from fls.flbase import MODEL
from Solver.co_opt import CO_OPT, EFO_OPT
from Solver.mco_opt import MCO_OPT
import torch
import numpy as np
from torch.autograd import Variable
from Solver.min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.optim.lr_scheduler import StepLR
import copy
from dataload import Load_Initial_Dataset
import random
import time


class EFFL(MODEL):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.co_opt = CO_OPT(n=self.n_params, eps_delta=None)
        self.eps = args.eps
        self.scheduler = StepLR(self.optim, step_size=500, gamma=0.2)

    def train_stage1(self, epoch):
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.step_size,
            momentum=0.0,
            weight_decay=1e-4,
        )

        self.model.train()
        self.optim.zero_grad()

        grads_performance = []
        grads_disparity = []
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
        client_disparities_ori = []

        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx]
                )
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)

            ############################################################## GPU version
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )
            grad = torch.stack(grad)

            grads_performance.append(grad)
            self.optim.zero_grad()

            torch.abs(pred_dis).backward(retain_graph=True)

            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )

            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )

            grad = torch.stack(grad)
            grads_disparity.append(grad)
            self.optim.zero_grad()

            client_disparities.append(torch.abs(pred_dis))
            client_disparities_ori.append(
                torch.abs(torch.tensor(dis)).to(pred_dis.device)
            )
            client_losses.append(loss)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)

       
        # mean
        alphas_l = torch.from_numpy(
            np.ones((1, len(grads_performance))) / len(grads_performance)
        ).to(client_losses[0].device)
        alphas_l = alphas_l.float()
        alphas_l = alphas_l.view(1, -1)
        grad_l = alphas_l @ torch.stack(grads_performance)

        max_performance_loss = torch.max(torch.stack(client_losses)).item()

        alphas_g = torch.from_numpy(
            np.ones((1, len(grads_disparity))) / len(grads_disparity)
        ).to(client_losses[0].device)

        # max
        # max_disparity_loss_index = torch.argmax(torch.stack(client_disparities)).item()
        # alphas_g = torch.from_numpy(np.zeros((1, len(grads_performance)))).to(
        #     client_losses[0].device
        # )
        # alphas_g[0, max_disparity_loss_index] = 1

        alphas_g = alphas_g.float()

        max_disparity_loss = torch.max(torch.stack(client_disparities)).item()

        alphas_g = alphas_g.view(1, -1)
        grad_g = alphas_g @ torch.stack(grads_disparity)

        weighted_loss1 = torch.sum(torch.stack(client_losses) * alphas_l)
        weighted_loss2 = torch.sum(torch.stack(client_disparities) * alphas_g)
        if grad_g.shape[0] != 1:
            print(grad_g.shape)
        # cal weight between goal 1 and 3
        start_time = time.time()
        alpha_performance, _ = self.co_opt.get_alpha(
            [max_performance_loss, max_disparity_loss],
            [grad_l, grad_g],
            None,
            self.eps[0],
            None,
            None,
            goal_index=[0, 1],
        )
        end_time = time.time()
        self.logger.info(
            "stage1_gradient_single_runtime: {}".format(end_time - start_time)
        )
        if torch.cuda.is_available():
            alpha_performance = torch.from_numpy(alpha_performance.reshape(-1)).cuda()
        else:
            alpha_performance = torch.from_numpy(alpha_performance.reshape(-1))
        alpha_performance = alpha_performance.view(-1)

        self.logger.info(
            " epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {},  alpha_performance: {}".format(
                self.global_epoch,
                losses_data,
                pred_disparities_data,
                disparities_data,
                accs_data,
                alpha_performance,
            )
        )

        self.optim.zero_grad()
        weighted_loss = torch.sum(
            torch.stack([weighted_loss1, weighted_loss2]) * alpha_performance
        )
        weighted_loss.backward()
        self.optim.step()
        self.scheduler.step()

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.local_valid_stage()
            losses, accs, client_disparities, pred_dis, aucs = self.global_valid_stage()

        self.global_epoch += 1

    def train_stage2(self, epoch):
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.step_size,
            momentum=0.0,
            weight_decay=1e-4,
        )

        self.model.train()
        self.optim.zero_grad()

        grads_performance = []
        grads_disparity = []
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
        client_disparities_ori = []

        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx]
                )
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)

            ############################################################## GPU version
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )

            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )

            grad = torch.stack(grad)
            grads_performance.append(grad)
            self.optim.zero_grad()

            torch.abs(pred_dis).backward(retain_graph=True)

            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )
            grad = torch.stack(grad)
            grads_disparity.append(grad)
            self.optim.zero_grad()

            client_disparities.append(torch.abs(pred_dis))
            client_disparities_ori.append(
                torch.abs(torch.tensor(dis)).to(pred_dis.device)
            )
            client_losses.append(loss)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)

        # Calculate the gradient and loss of the absolute value difference
        grads_performance_variance = []
        client_losses_variance = []
        grads_disparity_variance = []
        client_disparities_variance = []

        means_loss = torch.mean(torch.stack(client_losses))
        mean_disparity = torch.mean(torch.stack(client_disparities))
        mean_grad_performance = torch.mean(torch.stack(grads_performance))
        mean_grad_disparity = torch.mean(torch.stack(grads_disparity))

        for client_idx in range(self.n_clients):
            client_loss_variance = torch.abs(client_losses[client_idx] - means_loss)
            client_loss_variance.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            grad = torch.stack(grad)

            if self.attack_type == "Random" or "Enlarge" or "Zero":
                if self.random_list[client_idx] < self.attack_ratio:
                    grad = Variable(
                        torch.sgn(client_losses[client_idx] - means_loss)
                        * (grads_performance[client_idx] - mean_grad_performance),
                        requires_grad=False,
                    )
            grads_performance_variance.append(grad)
            client_losses_variance.append(client_loss_variance)
            self.optim.zero_grad()

            client_disparity_variance = torch.abs(
                client_disparities[client_idx] - mean_disparity
            )
            client_disparity_variance.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            grad = torch.stack(grad)

            if self.attack_type == "Random" or "Enlarge" or "Zero":
                if self.random_list[client_idx] < self.attack_ratio:
                    grad = Variable(
                        torch.sgn(client_disparities[client_idx] - mean_disparity)
                        * (grads_disparity[client_idx] - mean_grad_disparity),
                        requires_grad=False,
                    )
            grads_disparity_variance.append(grad)
            client_disparities_variance.append(client_disparity_variance)
            self.optim.zero_grad()

       
        alphas_l = torch.from_numpy(
            np.ones((1, len(grads_performance))) / len(grads_performance)
        ).to(client_losses[0].device)
        alphas_l = alphas_l.float()
        max_performance_loss = torch.max(torch.stack(client_losses)).item()

        alphas_g = torch.from_numpy(
            np.ones((1, len(grads_disparity))) / len(grads_disparity)
        ).to(client_losses[0].device)

        # max
        # max_disparity_loss_index = torch.argmax(torch.stack(client_disparities)).item()
        # alphas_g = torch.from_numpy(np.zeros((1, len(grads_performance)))).to(
        #     client_losses[0].device
        # )
        # alphas_g[0, max_disparity_loss_index] = 1

        alphas_g = alphas_g.float()
        max_disparity_loss = torch.max(torch.stack(client_disparities)).item()

        # mean
        alphas_vl = torch.from_numpy(
            np.ones((1, len(grads_performance))) / len(grads_performance)
        ).to(client_losses[0].device)

        # max
        # max_client_losses_variance_index = torch.argmax(
        #     torch.stack(client_losses_variance)
        # ).item()
        # alphas_vl = torch.from_numpy(np.zeros((1, len(client_losses_variance)))).to(
        #     client_losses_variance[0].device
        # )
        # alphas_vl[0, max_client_losses_variance_index] = 1

        alphas_vl = alphas_vl.float()
        variance_max_performance = torch.max(torch.stack(client_losses_variance)).item()

        # mean
        alphas_vg = torch.from_numpy(
            np.ones((1, len(grads_disparity))) / len(grads_disparity)
        ).to(client_losses[0].device)
        # max
        # max_client_disparities_variance_index = torch.argmax(
        #     torch.stack(client_disparities_variance)
        # ).item()
        # alphas_vg = torch.from_numpy(
        #     np.zeros((1, len(client_disparities_variance)))
        # ).to(client_disparities_variance[0].device)
        # alphas_vg[0, max_client_disparities_variance_index] = 1

        alphas_vg = alphas_vg.float()
        variance_max_disparity = torch.max(
            torch.stack(client_disparities_variance)
        ).item()

        # mean
        alphas_l = alphas_l.view(1, -1)
        grad_l = alphas_l @ torch.stack(grads_performance)
        # mean
        alphas_g = alphas_g.view(1, -1)
        grad_g = alphas_g @ torch.stack(grads_disparity)
        alphas_vl = alphas_vl.view(1, -1)
        grad_vl = alphas_vl @ torch.stack(grads_performance_variance)
        alphas_vg = alphas_vg.view(1, -1)
        grad_vg = alphas_vg @ torch.stack(grads_disparity_variance)

        weighted_loss1 = torch.sum(torch.stack(client_losses) * alphas_l)
        weighted_loss2 = torch.sum(torch.stack(client_disparities) * alphas_g)
        weighted_loss3 = torch.sum(torch.stack(client_losses_variance) * alphas_vl)
        weighted_loss4 = torch.sum(torch.stack(client_disparities_variance) * alphas_vg)

        if max_disparity_loss <= self.eps[0]:
            grad_g = torch.zeros_like(grad_g, requires_grad=False)

        if variance_max_performance <= self.eps[1]:
            grad_vl = torch.zeros_like(grad_vl, requires_grad=False)

        if variance_max_disparity <= self.eps[2]:
            grad_vg = torch.zeros_like(grad_vg, requires_grad=False)
        ###################  Normalized  ##############################
        gn = gradient_normalizers(
            [grad_vl, grad_vg, grad_l, grad_g],
            [weighted_loss3, weighted_loss4, weighted_loss1, weighted_loss2],
            "l2",
        )

        grad_vl = grad_vl / gn[0]
        grad_vg = grad_vg / gn[1]
        grad_l = grad_l / gn[2]
        grad_g = grad_g / gn[3]
        ##############################################################
        if variance_max_performance > self.eps[1]:
            grads = torch.cat(
                (
                    grad_vl,
                    grad_vg,
                    grad_l,
                    grad_g,
                ),
                dim=0,
            )

            grad_vl = grad_vl.t()
            ###
            start_time = time.time()
            alpha, gamma = self.mco_opt.get_alpha(grads, grad_vl, grads.t())

        elif variance_max_disparity > self.eps[2]:
            grads = torch.cat(
                (
                    grad_vg,
                    grad_vl,
                    grad_l,
                    grad_g,
                ),
                dim=0,
            )

            grad_vg = grad_vg.t()
            ###
            start_time = time.time()
            alpha, gamma = self.mco_opt.get_alpha(grads, grad_vg, grads.t())
        else:
            grads = torch.cat(
                (
                    grad_l,
                    grad_vg,
                    grad_vl,
                    grad_g,
                ),
                dim=0,
            )

            grad_l = grad_l.t()
            ###
            start_time = time.time()
            alpha, gamma = self.mco_opt.get_alpha(grads, grad_l, grads.t())

        end_time = time.time()
        self.logger.info(
            "stage2_gradient_single_runtime: {}".format(end_time - start_time)
        )
        if torch.cuda.is_available():
            alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
        else:
            alpha = torch.from_numpy(alpha.reshape(-1))
        if variance_max_performance > self.eps[1]:
            considered_loss = [weighted_loss3]
            considered_loss.append(weighted_loss4)
            considered_loss.append(weighted_loss1)
            considered_loss.append(weighted_loss2)
        elif variance_max_disparity > self.eps[2]:
            considered_loss = [weighted_loss4]
            considered_loss.append(weighted_loss3)
            considered_loss.append(weighted_loss1)
            considered_loss.append(weighted_loss2)
        else:
            considered_loss = [weighted_loss1]
            considered_loss.append(weighted_loss4)
            considered_loss.append(weighted_loss3)
            considered_loss.append(weighted_loss2)
        weighted_loss = torch.sum(torch.stack(considered_loss) * alpha)
        weighted_loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.logger.info(
            "1, epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {},  alphas:{}".format(
                self.global_epoch,
                losses_data,
                pred_disparities_data,
                disparities_data,
                accs_data,
                alpha,
            )
        )

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.local_valid_stage()
            losses, accs, client_disparities, pred_dis, aucs = self.global_valid_stage()
        self.global_epoch += 1

    def train_stage3(self, epoch, worse=True):
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.step_size,
            momentum=0.0,
            weight_decay=1e-4,
        )

        self.model.train()
        self.optim.zero_grad()

        grads_performance = []
        grads_disparity = []
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []
        client_disparities_ori = []

        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx]
                )
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y = self.model(X, Y, A)

            ############################################################## GPU version
            loss.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )

            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )

            grad = torch.stack(grad)
            grads_performance.append(grad)
            self.optim.zero_grad()

            torch.abs(pred_dis).backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            if self.attack_type != None:
                if self.random_list[client_idx] < self.attack_ratio:
                    if self.attack_type == "Random":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                new_grad = torch.rand_like(
                                    param.grad.data.clone().flatten()
                                )
                                new_grad = new_grad - new_grad.mean()

                                grad.extend(
                                    Variable(
                                        new_grad,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Enlarge":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 10,
                                        requires_grad=False,
                                    )
                                )
                    if self.attack_type == "Zero":
                        grad = []
                        # Maliciousclients transmit random-zero-mean gradient
                        for param in self.model.parameters():
                            if param.grad is not None:
                                grad.extend(
                                    Variable(
                                        param.grad.data.clone().flatten() * 0.001,
                                        requires_grad=False,
                                    )
                                )
            grad = torch.stack(grad)
            grads_disparity.append(grad)
            self.optim.zero_grad()

            client_disparities.append(torch.abs(pred_dis))
            client_disparities_ori.append(
                torch.abs(torch.tensor(dis)).to(pred_dis.device)
            )
            client_losses.append(loss)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)

        # Calculate the gradient and loss of the absolute value difference
        grads_performance_variance = []
        client_losses_variance = []
        grads_disparity_variance = []
        client_disparities_variance = []

        means_loss = torch.mean(torch.stack(client_losses))
        mean_disparity = torch.mean(torch.stack(client_disparities))
        mean_grad_performance = torch.mean(torch.stack(grads_performance))
        mean_grad_disparity = torch.mean(torch.stack(grads_disparity))

        for client_idx in range(self.n_clients):
            client_loss_variance = torch.abs(client_losses[client_idx] - means_loss)
            client_loss_variance.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            grad = torch.stack(grad)
            if self.attack_type == "Random" or "Enlarge" or "Zero":
                if self.random_list[client_idx] < self.attack_ratio:
                    grad = Variable(
                        torch.sgn(client_losses[client_idx] - means_loss)
                        * (grads_performance[client_idx] - mean_grad_performance),
                        requires_grad=False,
                    )

            grads_performance_variance.append(grad)
            client_losses_variance.append(client_loss_variance)
            self.optim.zero_grad()

            client_disparity_variance = torch.abs(
                client_disparities[client_idx] - mean_disparity
            )
            client_disparity_variance.backward(retain_graph=True)
            grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grad.extend(
                        Variable(param.grad.data.clone().flatten(), requires_grad=False)
                    )
            grad = torch.stack(grad)
            if self.attack_type == "Random" or "Enlarge" or "Zero":
                if self.random_list[client_idx] < self.attack_ratio:
                    grad = Variable(
                        torch.sgn(client_disparities[client_idx] - mean_disparity)
                        * (grads_disparity[client_idx] - mean_grad_disparity),
                        requires_grad=False,
                    )
            grads_disparity_variance.append(grad)
            client_disparities_variance.append(client_disparity_variance)
            self.optim.zero_grad()

        
        alphas_l = torch.from_numpy(
            np.ones((1, len(grads_performance))) / len(grads_performance)
        ).to(client_losses[0].device)

        alphas_l = alphas_l.float()
        max_performance_loss = torch.mean(torch.stack(client_losses)).item()

        alphas_g = torch.from_numpy(
            np.ones((1, len(grads_disparity))) / len(grads_disparity)
        ).to(client_losses[0].device)
        # max
        # max_client_disparities_index = torch.argmax(
        #     torch.stack(client_disparities)
        # ).item()
        # alphas_g = torch.from_numpy(np.zeros((1, len(client_disparities)))).to(
        #     client_disparities[0].device
        # )
        # alphas_g[0, max_client_disparities_index] = 1

        alphas_g = alphas_g.float()
        max_disparity_loss = torch.max(torch.stack(client_disparities)).item()

        alphas_vl = torch.from_numpy(
            np.ones((1, len(grads_performance))) / len(grads_performance)
        ).to(client_losses[0].device)
        # max
        # max_client_losses_variance_index = torch.argmax(
        #     torch.stack(client_losses_variance)
        # ).item()
        # alphas_vl = torch.from_numpy(np.zeros((1, len(client_losses_variance)))).to(
        #     client_losses_variance[0].device
        # )
        # alphas_vl[0, max_client_losses_variance_index] = 1

        alphas_vl = alphas_vl.float()
        variance_max_performance = torch.max(torch.stack(client_losses_variance)).item()

        alphas_vg = torch.from_numpy(
            np.ones((1, len(grads_disparity))) / len(grads_disparity)
        ).to(client_losses[0].device)

        # max
        # max_client_disparitie_variance_index = torch.argmax(
        #     torch.stack(client_disparities_variance)
        # ).item()
        # alphas_vg = torch.from_numpy(
        #     np.zeros((1, len(client_disparities_variance)))
        # ).to(client_disparities_variance[0].device)
        # alphas_vg[0, max_client_disparitie_variance_index] = 1

        alphas_vg = alphas_vg.float()
        variance_max_disparity = torch.max(
            torch.stack(client_disparities_variance)
        ).item()

        client_pred_disparity = torch.sum(alphas_g * torch.stack(client_disparities))
        client_loss_variance = torch.sum(
            alphas_vl * torch.stack(client_losses_variance)
        )
        client_disparity_variance = torch.sum(
            alphas_vg * torch.stack(client_disparities_variance)
        )

        alphas_l = alphas_l.view(1, -1)
        grad_l = alphas_l @ torch.stack(grads_performance)
        alphas_g = alphas_g.view(1, -1)
        grad_g = alphas_g @ torch.stack(grads_disparity)
        alphas_vl = alphas_vl.view(1, -1)
        grad_vl = alphas_vl @ torch.stack(grads_performance_variance)
        alphas_vg = alphas_vg.view(1, -1)
        grad_vg = alphas_vg @ torch.stack(grads_disparity_variance)

        if max_disparity_loss < self.eps[0]:
            grad_g = torch.zeros_like(grad_g, requires_grad=False)

        if variance_max_performance < self.eps[1]:
            grad_vl = torch.zeros_like(grad_vl, requires_grad=False)

        if variance_max_disparity < self.eps[2]:
            grad_vg = torch.zeros_like(grad_vg, requires_grad=False)
        ####################  worse update ###############################
        if worse == True:
            client_losses_worse = []
            grads_performance_worse = []
            client_losses_better = []
            grads_performance_better = []

            for i in range(len(client_losses)):
                if client_losses[i] > max_performance_loss:
                    client_losses_worse.append(client_losses[i])
                    grads_performance_worse.append(grads_performance[i])
                else:
                    client_losses_better.append(client_losses[i])
                    grads_performance_better.append(grads_performance[i])

            grads_performance_worse = torch.stack(grads_performance_worse)
            grads_performance_better = torch.stack(grads_performance_better)
            self.mco_opt = MCO_OPT(
                n_theta=self.n_params, n_alpha=3 + len(client_losses_worse)
            )

            grad_performance = torch.mean(grads_performance_worse, dim=0, keepdim=True)

            grads = torch.cat(
                (
                    grads_performance_worse,
                    # grads_performance_better,
                    grad_g,
                    grad_vl,
                    grad_vg,
                ),
                dim=0,
            )

            grad_performance = grad_performance.t()

            alpha, gamma = self.mco_opt.get_alpha(grads, grad_performance, grads.t())
            if torch.cuda.is_available():
                alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
            else:
                alpha = torch.from_numpy(alpha.reshape(-1))

            # worse
            client_losses_worse.append(client_pred_disparity)
            client_losses_worse.append(client_loss_variance)
            client_losses_worse.append(client_disparity_variance)
            weighted_loss = torch.sum(torch.stack(client_losses_worse) * alpha)
        #############################################################
        if worse == False:
            ###################  Normalized for eicu ##############################
            if "eicu" in self.dataset:
                gn = gradient_normalizers(
                    grads_performance + [grad_g, grad_vl, grad_vg],
                    None,
                    "l2",
                )

                for g_i in range(len(grads_performance)):
                    grads_performance[g_i] /= gn[g_i]

                grad_g = grad_g / gn[-3]
                grad_vl = grad_vl / gn[-2]
                grad_vg = grad_vg / gn[-1]
            ##############################################################
            # grads_performance = torch.stack(grads_performance)
            # grad_performance = torch.mean(grads_performance, dim=0, keepdim=True)

            # grads = torch.cat(
            #     (
            #         grads_performance,
            #         grad_g,
            #         grad_vl,
            #         grad_vg,
            #     ),
            #     dim=0,
            # )
            # grad_performance = grad_performance.t()

            # alpha, gamma = self.mco_opt.get_alpha(grads, grad_performance, grads.t())
            #################max#####################################
            max_index = client_losses.index(max(client_losses))
            grads_performance = torch.stack(grads_performance)
            grad_performance = grads_performance[max_index].unsqueeze(0)
            grads = torch.cat(
                (
                    grads_performance,
                    grad_g,
                    grad_vl,
                    grad_vg,
                ),
                dim=0,
            )
            grad_performance = grad_performance.t()

            start_time = time.time()
            alpha, gamma = self.mco_opt.get_alpha(grads, grad_performance, grads.t())
            end_time = time.time()
            self.logger.info(
                "stage3_gradient_single_runtime: {}".format(end_time - start_time)
            )
            ####################################################################
            if torch.cuda.is_available():
                alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
            else:
                alpha = torch.from_numpy(alpha.reshape(-1))

            # no-worse
            client_losses.append(client_pred_disparity)
            client_losses.append(client_loss_variance)
            client_losses.append(client_disparity_variance)
            weighted_loss = torch.sum(torch.stack(client_losses) * alpha)
        ###############################################################

        weighted_loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.logger.info(
            "1, epoch: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {},alphas:{}".format(
                self.global_epoch,
                losses_data,
                pred_disparities_data,
                disparities_data,
                accs_data,
                alpha,
            )
        )

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.local_valid_stage()
            losses, accs, client_disparities, pred_dis, aucs = self.global_valid_stage()
        self.global_epoch += 1

    def train(self):
        self.random_list = [random.uniform(0, 1) for i in range(self.n_clients)]

        if len(self.max_epoch) != 3:
            raise ValueError("epoch must contain 3 stages")
        start_epoch = self.global_epoch
        # stage 1
        start_time = time.time()
        for epoch in range(start_epoch, self.max_epoch[0]):
            self.train_stage1(epoch)
        end_time = time.time()
        self.logger.info("stage1_runtime: {}".format(end_time - start_time))
        # stage 2
        start_time = time.time()
        self.mco_opt = MCO_OPT(n_theta=self.n_params, n_alpha=4)
        for epoch in range(self.max_epoch[0], self.max_epoch[0] + self.max_epoch[1]):
            self.train_stage2(epoch)
        end_time = time.time()
        self.logger.info("stage2_runtime: {}".format(end_time - start_time))
        # stage 3
        start_time = time.time()
        self.mco_opt = MCO_OPT(n_theta=self.n_params, n_alpha=3 + self.n_clients)
        for epoch in range(
            self.max_epoch[0] + self.max_epoch[1],
            self.max_epoch[0] + self.max_epoch[1] + self.max_epoch[2],
        ):
            self.train_stage3(epoch, worse=False)
        end_time = time.time()
        self.logger.info("stage3_runtime: {}".format(end_time - start_time))

    def dynamic_train(self, args):
        self.client_train_loaders, self.client_test_loaders = Load_Initial_Dataset(args)

        print("end")
