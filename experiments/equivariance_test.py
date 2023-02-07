import os
import GPUtil
import copy
import unittest
import hydra
import torch
import tree
import numpy as np

from scipy.spatial.transform import Rotation
from scipy import stats
from experiments import train_se3_diffusion
from openfold.data import data_transforms
from openfold.utils import rigid_utils as ru
from data import utils as du


class EquivarianceTest(unittest.TestCase):

    def setUp(self):

        chosen_gpu = ''.join([
            str(x) for x in GPUtil.getAvailable(order='memory')])
        print(f'Running on GPU: {chosen_gpu}')
        self.device = f"cuda:{chosen_gpu}" 
        torch.cuda.set_device(self.device)

        with hydra.initialize(version_base=None, config_path="../config"):
            self.conf = hydra.compose(
                config_name="base", overrides=[
                    "data.rosetta.filtering.subset=1",
                ])

    def sample_feats_with_and_without_rotation(self, train_loader):
        """sample_feats_with_and_without_rotation samples two examples that are the same up to a rotation.

        Returns:
            the two examples and the function to rotate.
        """
        # Initialize single data example.
        data_iter = iter(train_loader)
        raw_data_feats = next(data_iter)
        batch_idx = 0
        data_feats = tree.map_structure(
            lambda x: x[batch_idx], raw_data_feats)
        res_mask = data_feats['res_mask']
        init_feats = tree.map_structure(
            lambda x: x[res_mask.bool()] if x.ndim > 0 else x, data_feats)
        init_feats = tree.map_structure(lambda x: x[None], init_feats)

        # Set the input rigids to the true rigids. We do this because the
        # rotation will be performed on the ground truth atoms that constructed
        # rigids_0.
        init_feats['rigids_t'] = init_feats['rigids_0']

        # Rotate ground truth atoms.
        rot = torch.tensor(Rotation.from_euler('z', 90, degrees=True).as_matrix()).double()
        rot_pos_fn = lambda x: torch.einsum('...i,ji->...j', x, rot) 
        atom_pos_rot = torch.clone(init_feats['atom37_pos'])
        atom_pos_rot = rot_pos_fn(atom_pos_rot)
        chain_feats = {
            'aatype': init_feats['aatype'][0],
            'all_atom_positions': atom_pos_rot[0],
            'all_atom_mask': init_feats['atom37_mask'][0]
        }
        rigids_rot = data_transforms.atom37_to_frames(chain_feats)['rigidgroups_gt_frames'][:, 0]
        rigids_rot = ru.Rigid.from_tensor_4x4(rigids_rot)
        rigids_rot = rigids_rot.apply_trans_fn(
            lambda x: x/self.conf.data.rosetta.scale_factor)

        # Constructed rotated data.
        perturbed_feats = copy.deepcopy(init_feats)
        perturbed_feats['rigids_t'] = rigids_rot.to_tensor_7()[None]
        perturbed_feats['rigids_0'] = rigids_rot.to_tensor_7()[None]
        
        # Return initial and perturbed features
        return init_feats, perturbed_feats, rot_pos_fn


    def test_equivariance(self):
        # Initialize experiment
        exp = train_se3_diffusion.Experiment(conf=self.conf)
        exp._model = exp._model.to(self.device)
        train_loader, _ = exp.create_rosetta_dataset(0, 1)

        # Briefly train model s.t. predictions are not zero.
        print('Begin mini training.')
        train_steps = 100
        step = 0
        exp._model.train()
        while step <= train_steps:
            for train_features in train_loader:
                step += 1
                exp.step = step
                train_features = tree.map_structure(
                    lambda x: x.to(self.device), train_features)
                exp.update_fn(train_features)
                if step > train_steps:
                    break
        print('Finished mini training.')

        init_feats, perturbed_feats, rot_pos_fn = self.sample_feats_with_and_without_rotation(train_loader)


        #########################################################
        # Test that predicted scores are invariant/equivariant. #
        #########################################################
        to_device = lambda y: tree.map_structure(lambda x: x.to(self.device), y)
        init_out = exp.model(to_device(init_feats))
        pertub_out = exp.model(to_device(perturbed_feats))

        self.assertTrue(torch.all(init_out['rot_score'] != 0))
        self.assertTrue(torch.all(pertub_out['rot_score'] != 0))

        rot_score = init_out['rot_score'].double().cpu() 
        if self.conf.model.equivariant_rot_score:
            rot_score = rot_pos_fn(rot_score)

        np.testing.assert_allclose(
            du.move_to_np(rot_score),
            du.move_to_np(pertub_out['rot_score']), 
            atol=0.01
        )

        rot_trans_score = rot_pos_fn(init_out['trans_score'].double().cpu())
        np.testing.assert_allclose(
            du.move_to_np(rot_trans_score),
            du.move_to_np(pertub_out['trans_score']), 
            atol=0.01
        )

        print('Passed: single step.')

        ################################################
        # Test that samples are invariant/equivariant. #
        ################################################

        # Run inference
        infer_out = exp.inference_fn(
            to_device(init_feats), add_noise=False)
        perturbed_out = exp.inference_fn(
            to_device(perturbed_feats), add_noise=False)

        sample_prots = infer_out[1]
        perturb_prots = perturbed_out[1]

        sample_prots = torch.stack(sample_prots).cpu()
        perturb_prots = torch.stack(perturb_prots).cpu()
        rot_sample_prots = rot_pos_fn(sample_prots.double())

        atom_pos37_mask = init_feats['atom37_mask'].cpu()[None, ..., None]
        rot_sample_prots = rot_sample_prots * atom_pos37_mask
        perturb_prots = perturb_prots * atom_pos37_mask
        np.testing.assert_allclose(
            du.move_to_np(rot_sample_prots),
            du.move_to_np(perturb_prots), 
            atol=0.05
        )

        print('Passed: reverse diffusion.')

        ################################################
        # Test that computed likelihoods are invariant
        ################################################
        init_feats = tree.map_structure(lambda x: x[0], init_feats)
        perturbed_feats = tree.map_structure(lambda x: x[0], perturbed_feats)

        n_samples = 8
        p_val_thresh = 0.05
        init_lls, perturb_lls = [], []
        init_rigids_0, init_ll = exp.log_likelihood(to_device(init_feats))
        perturb_rigids_0, perturb_ll = exp.log_likelihood(to_device(perturbed_feats))

        # Compare init and perturbed translations
        init_trans  = ru.Rigid.from_tensor_7(init_rigids_0).get_trans().cpu().double()
        perturbed_trans  = ru.Rigid.from_tensor_7(perturb_rigids_0).get_trans().cpu().double()
        init_trans_rot = rot_pos_fn(init_trans)
        np.testing.assert_allclose(
            du.move_to_np(init_trans_rot),
            du.move_to_np(perturbed_trans), 
            atol=0.01
        )
        print('Passed: rotationally equivariant embeddings.')

        init_lls.append(float(init_ll.detach().cpu()))
        perturb_lls.append(float(perturb_ll.detach().cpu()))

        # Because likelihoods are stochastically computed, we compare the 
        # distributions of computed likelihoods with a ttest.
        for i in range(1, n_samples):
            _, init_ll = exp.log_likelihood(to_device(init_feats))
            _, perturb_ll = exp.log_likelihood(to_device(perturbed_feats))
            init_lls.append(float(init_ll.detach().cpu()))
            perturb_lls.append(float(perturb_ll.detach().cpu().numpy()))

        init_lls, perturb_lls = np.array(init_lls), np.array(perturb_lls)
        ttest_pvalue = stats.ttest_ind(
            init_lls,
            perturb_lls
        ).pvalue
        np.testing.assert_(ttest_pvalue > p_val_thresh, "p-value: " + str(ttest_pvalue))
        print('Passed: likelihood distributions are not significantly different')






if __name__ == '__main__':
    unittest.main()