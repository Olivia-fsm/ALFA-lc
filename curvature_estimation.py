"Estimate model curvature using the power method"

import os
import logging
from typing import Tuple
import argparse
import datetime as dt
import random

import torch
import torch.nn.functional as F


# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def curvature_hessian_estimator(model: torch.nn.Module,
                        image: torch.Tensor,
                        target: torch.Tensor,
                        num_power_iter: int=10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    model.eval()
    u = torch.randn_like(image)
    u /= torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)

    with torch.enable_grad():
        image = image.requires_grad_()
        out = model(image)
        y = F.log_softmax(out, 1)
        output = F.nll_loss(y, target, reduction='none')
        model.zero_grad()
        # Gradients w.r.t. input
        gradients = torch.autograd.grad(outputs=output.sum(),
                                        inputs=image, create_graph=True)[0]
        gnorm = torch.norm(gradients, p=2, dim=(1, 2, 3))
        assert not gradients.isnan().any()

        # Power method to find singular value of Hessian
        for _ in range(num_power_iter):
            grad_vector_prod = (gradients * u.detach_()).sum()
            hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image, retain_graph=True)[0]
            assert not hessian_vector_prod.isnan().any()

            hvp_norm = torch.norm(hessian_vector_prod, p=2, dim=(1, 2, 3), keepdim=True)
            u = hessian_vector_prod.div(hvp_norm + 1e-6) #1e-6 for numerical stability

        grad_vector_prod = (gradients * u.detach_()).sum()
        hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image)[0]
        hessian_singular_value = (hessian_vector_prod * u.detach_()).sum((1, 2, 3))
    
    # curvature = hessian_singular_value / (grad_norm + epsilon) by definition
    curvatures = hessian_singular_value.abs().div(gnorm + 1e-6)
    hess = hessian_singular_value.abs()
    grad = gnorm
    
    return curvatures, hess, grad


def measure_curvature(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      data_fraction: float=1.0,
                      batch_size: int=8,
                      num_power_iter: int=10,
                      device: torch.device='cpu'):

    """
    Compute curvature, hessian norm and gradient norm of a subset of the data given by the dataloader.
    These values are computed using the power method, which requires setting the number of power iterations (num_power_iter).
    """

    model.eval()
    datasize = int(data_fraction * len(dataloader.dataset))
    max_batches = int(datasize / batch_size)
    curvature_agg = torch.zeros(size=(max_batches*batch_size,))
    grad_agg = torch.zeros(size=(max_batches*batch_size,))
    hess_agg = torch.zeros(size=(max_batches*batch_size,))

    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        with torch.no_grad():
            curvatures, hess, grad = curvature_hessian_estimator(model, data, target, num_power_iter=num_power_iter)
        try:
            curvature_agg[idx * batch_size:(idx + 1) * batch_size] = curvatures.detach()
            hess_agg[idx * batch_size:(idx + 1) * batch_size] = hess.detach()
            grad_agg[idx * batch_size:(idx + 1) * batch_size] = grad.detach()
        except:
            print('curvatures', curvatures.shape)
            print('hess', hess.shape)
            print('grad', grad.shape)
            print('curvature_agg', curvature_agg.shape)
        

        avg_curvature, std_curvature = curvature_agg.mean().item(), curvature_agg.std().item()
        avg_hessian, std_hessian = hess_agg.mean().item(), hess_agg.std().item()
        avg_grad, std_grad = grad_agg.mean().item(), grad_agg.std().item()

        if idx == (max_batches - 1):
            logger.info('Average Curvature: {:.6f} +/- {:.2f} '.format(avg_curvature, std_curvature))
            logger.info('Average Hessian Spectral Norm: {:.6f} +/- {:.2f} '.format(avg_hessian, std_hessian))
            logger.info('Average Gradient Norm: {:.6f} +/- {:.2f}'.format(avg_grad, std_grad))
            return {
                'curvature_avg': avg_curvature,
                'curvature_std': std_curvature,
                'hessian_avg': avg_hessian,
                'hessian_std': std_hessian,
                'grad_avg': avg_grad,
                'grad_std': std_grad,
            }
        
    return {
                'curvature_avg': avg_curvature,
                'curvature_std': std_curvature,
                'hessian_avg': avg_hessian,
                'hessian_std': std_hessian,
                'grad_avg': avg_grad,
                'grad_std': std_grad,
            }


# if __name__ == "__main__":
#     args = main()
#     model, train_loader, test_loader, device = get_model_and_datasets(args)

#     logger.info("\nEstimating curvature on training data...")
#     measure_curvature(model, train_loader, 
#                         data_fraction=args.data_fraction, 
#                         batch_size=args.batch_size, 
#                         num_power_iter=args.num_power_iter,
#                         device=device)

#     logger.info("\nEstimating curvature on test data...")
#     measure_curvature(model, test_loader, 
#                         data_fraction=args.data_fraction, 
#                         batch_size=args.batch_size, 
#                         num_power_iter=args.num_power_iter,
#                         device=device)