# lenet base model for Pareto MTL
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataload import LoadDataset
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import statistics
from models import RegressionModel
import random
import copy


class NNTrain(torch.nn.Module):
    def __init__(self, model, disparity_type="DP", dataset="adult"):
        super(NNTrain, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        self.disparity_type = disparity_type
        self.dataset = dataset
        self.create_Loc_reshape_list()

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, params):
        self.model.load_state_dict(params)

    def span_model_params_to_vec(self):
        param_vec = []
        model_params = self.state_dict()
        for layer in model_params.keys():
            flat = model_params[layer].clone().flatten()
            param_vec.append(Variable(flat, requires_grad=False))
        param_vec = torch.cat(param_vec)
        return param_vec

    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.model.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(
                torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(
                    p.data.shape
                )
            )
            currentIdx += len(flat)

    def forward(self, x, y, A):
        ys_pre = self.model(x).flatten()
        ys = torch.sigmoid(ys_pre)
        hat_ys = (ys >= 0.5).float()

        task_loss = self.loss(ys, y)
        accs = torch.mean((hat_ys == y).float()).item()
        aucs = roc_auc_score(y.cpu(), ys.clone().detach().cpu())
        if True:
            if self.disparity_type == "DP":
                pred_dis = torch.sum(torch.sigmoid(50 * ys_pre) * A) / torch.sum(
                    A
                ) - torch.sum(torch.sigmoid(50 * ys_pre) * (1 - A)) / torch.sum(1 - A)
                disparitys = torch.sum(hat_ys * A) / torch.sum(A) - torch.sum(
                    hat_ys * (1 - A)
                ) / torch.sum(1 - A)
            elif self.disparity_type == "TPSD":
                group_ys_pre = []
                group_y = []
                group_hat_ys = []
                # find unique value in A and the corresponding index
                unique_values, _ = torch.unique(A, return_inverse=True)
                unique_indices = [
                    torch.nonzero(A == val).squeeze() for val in unique_values
                ]
                for i, A_value in enumerate(unique_values):
                    group_ys_pre.append(
                        torch.index_select(ys_pre, dim=0, index=unique_indices[i])
                    )
                    group_y.append(
                        torch.index_select(y, dim=0, index=unique_indices[i])
                    )
                    group_hat_ys.append(
                        torch.index_select(hat_ys, dim=0, index=unique_indices[i])
                    )
                # calculate pred_dis
                
                unique_values_index = []
                for i, A_value in enumerate(unique_values):
                    if torch.sum(group_y[i]) != 0:
                        unique_values_index.append(i)

                pred = [
                    torch.sum(torch.sigmoid(10 * group_ys_pre[i]) * group_y[i])
                    / torch.sum(group_y[i])
                    for i in unique_values_index
                ]
                mean_pred = torch.mean(torch.stack(pred), dim=0)
                pred_dis = torch.sqrt(
                    sum(
                        [
                            torch.pow(pred[i] - mean_pred, 2)
                            for i in range(len(unique_values_index))
                        ]
                    )
                    / len(unique_values_index)
                )
                # calculate disparitys
                disparity = [
                    torch.sum(group_hat_ys[i] * group_y[i]) / torch.sum(group_y[i])
                    for i in unique_values_index
                ]
                mean_disparitys = torch.mean(torch.stack(disparity), dim=0)
                disparitys = torch.sqrt(
                    sum(
                        [
                            torch.pow(disparity[i] - mean_disparitys, 2)
                            for i in range(len(unique_values_index))
                        ]
                    )
                    / len(unique_values_index)
                )

            elif self.disparity_type == "APSD":
                group_ys_pre = []
                group_y = []
                group_hat_ys = []
                # find unique value in A and the corresponding index
                unique_values, _ = torch.unique(A, return_inverse=True)

                unique_indices = [
                    torch.nonzero(A == val).squeeze() for val in unique_values
                ]

                # for i in range(len(unique_indices)):
                #     if unique_indices[i].ndim == 0:
                #         unique_indices[i] = unique_indices[i].unsqueeze(dim=0)
                for i, A_value in enumerate(unique_values):
                    group_ys_pre.append(
                        torch.index_select(ys_pre, dim=0, index=unique_indices[i])
                    )
                    group_y.append(
                        torch.index_select(y, dim=0, index=unique_indices[i])
                    )
                    group_hat_ys.append(
                        torch.index_select(hat_ys, dim=0, index=unique_indices[i])
                    )
                # calculate pred_dis
                pred = [
                    torch.sum(torch.sigmoid(10 * group_ys_pre[i]))
                    / len(unique_indices[i])
                    for i, A_value in enumerate(unique_values)
                ]
                mean_pred = torch.mean(torch.stack(pred), dim=0)
                pred_dis = torch.sqrt(
                    sum(
                        [
                            torch.pow(pred[i] - mean_pred, 2)
                            for i, A_value in enumerate(unique_values)
                        ]
                    )
                    / len(unique_values)
                )
                # calculate disparitys
                disparity = [
                    torch.sum(group_hat_ys[i]) / len(unique_indices[i])
                    for i, A_value in enumerate(unique_values)
                ]
                mean_disparitys = torch.mean(torch.stack(disparity), dim=0)
                disparitys = torch.sqrt(
                    sum(
                        [
                            torch.pow(disparity[i] - mean_disparitys, 2)
                            for i, A_value in enumerate(unique_values)
                        ]
                    )
                    / len(unique_values)
                )
            elif self.disparity_type == "Eoppo":
                if "eicu_d" in self.dataset:
                    pred_dis = torch.sum(
                        torch.sigmoid(10 * (1 - ys_pre)) * A * (1 - y)
                    ) / torch.sum(A * (1 - y)) - torch.sum(
                        torch.sigmoid(10 * (1 - ys_pre)) * (1 - A) * (1 - y)
                    ) / torch.sum(
                        (1 - A) * (1 - y)
                    )

                    disparitys = torch.sum((1 - hat_ys) * A * (1 - y)) / torch.sum(
                        A * (1 - y)
                    ) - torch.sum((1 - hat_ys) * (1 - A) * (1 - y)) / torch.sum(
                        (1 - A) * (1 - y)
                    )
                else:
                    pred_dis = torch.sum(torch.sigmoid(10 * ys_pre) * A * y) / (
                        torch.sum(A * y) + torch.tensor(1e-3).to(y.device).float()
                    ) - torch.sum(torch.sigmoid(10 * ys_pre) * (1 - A) * y) / torch.sum(
                        (1 - A) * y + torch.tensor(1e-3).to(y.device).float()
                    )
                    disparitys = torch.sum(hat_ys * A * y) / torch.sum(
                        A * y + torch.tensor(1e-3).to(y.device).float()
                    ) - torch.sum(hat_ys * (1 - A) * y) / torch.sum(
                        (1 - A) * y + torch.tensor(1e-3).to(y.device).float()
                    )
            disparitys = disparitys.item()
            return task_loss, accs, aucs, pred_dis, disparitys, hat_ys
        else:
            print("error model in forward")
            exit()

    def randomize(self):
        self.model.apply(weights_init)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.weight.data *= 0.1


class MODEL(object):
    def __init__(self, args, logger):
        super(MODEL, self).__init__()
        self.dataset = args.dataset
        self.norm = args.norm
        self.max_epoch = args.max_epoch_stage

        self.global_epoch = args.global_epoch
        self.log_pickle_dir = args.log_dir
        self.eval_epoch = args.eval_epoch

        self.model = NNTrain(
            # args.model,
            RegressionModel(args.n_feats, 0),
            args.disparity_type,
            args.dataset,
        )

        self.data_load(args)
        self.logger = logger
        self.logger.info(str(args))

        self.disparity_type = args.disparity_type

        self.sensitive_attr = args.sensitive_attr

        if torch.cuda.is_available():
            self.model.cuda()
        self.optim = torch.optim.SGD(
            self.model.parameters(), lr=args.step_size, momentum=0.0, weight_decay=1e-4
        )

        self.step_size = args.step_size

        _, n_params = self.getNumParams(self.model.parameters())
        self.n_params = n_params
        self.attack_type = args.attack_type
        self.attack_ratio = args.attack_ratio

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

    def data_load(self, args):
        self.client_train_loaders, self.client_test_loaders = LoadDataset(
            args, self.model.model
        )
        self.n_clients = len(self.client_train_loaders)
        self.iter_train_clients = [enumerate(i) for i in self.client_train_loaders]
        self.iter_test_clients = [enumerate(i) for i in self.client_test_loaders]

    def local_valid_stage(self):
        with torch.no_grad():
            losses = []
            accs = []
            diss = []
            pred_diss = []
            aucs = []

            loader = self.client_test_loaders
            for client_idx, client_test_loader in enumerate(loader):
                valid_loss = []
                valid_accs = []
                valid_diss = []
                valid_pred_dis = []
                valid_auc = []
                for it, (X, Y, A) in enumerate(client_test_loader):
                    X = X.float()
                    Y = Y.float()
                    A = A.float()
                    if torch.cuda.is_available():
                        X = X.cuda()
                        Y = Y.cuda()
                        A = A.cuda()

                    loss, acc, auc, pred_dis, disparity, pred_y = self.model(X, Y, A)
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
                    valid_diss.append(disparity)
                    valid_pred_dis.append(pred_dis.item())
                    valid_auc.append(auc)
                assert len(valid_auc) == 1
                losses.append(np.mean(valid_loss))
                accs.append(np.mean(valid_accs))
                diss.append(np.mean(valid_diss))
                pred_diss.append(np.mean(valid_pred_dis))
                aucs.append(np.mean(valid_auc))
        self.logger.info(
            "valid: True, epoch: {}, loss: {}, accuracy: {}, mean_accuracy:{},variance_accuracy:{}, disparity: {}, mean_disparity:{},variance_disparity:{}, pred_disparity: {}".format(
                self.global_epoch,
                losses,
                accs,
                statistics.mean(accs),
                statistics.mean(accs) - min(accs),
                [abs(x) for x in diss],
                statistics.mean([abs(x) for x in diss]),
                max([abs(x) for x in diss]) - statistics.mean([abs(x) for x in diss]),
                pred_diss,
            )
        )

        return losses, accs, diss, pred_diss, aucs

    def global_valid_stage(self, isPrint=True):
        with torch.no_grad():
            loader = self.client_test_loaders

            losses = []
            accs = []
            diss = []
            pred_diss = []
            aucs = []

            X, Y, A = None, None, None
            for client_idx, client_test_loader in enumerate(loader):
                for it, (X_new, Y_new, A_new) in enumerate(client_test_loader):
                    if X != None:
                        X = torch.concatenate((X, X_new))
                    else:
                        X = X_new
                    if Y != None:
                        Y = torch.concatenate((Y, Y_new))
                    else:
                        Y = Y_new
                    if A != None:
                        A = torch.concatenate((A, A_new))
                    else:
                        A = A_new
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, disparity, pred_y = self.model(X, Y, A)
            losses.append(loss.item())
            accs.append(acc)
            diss.append(disparity)
            pred_diss.append(pred_dis.item())
            aucs.append(auc)

            global_loss = np.mean(losses)
            global_acc = np.mean(accs)
            global_dis = np.mean(diss)
            global_pred_dis = np.mean(pred_diss)
            global_auc = np.mean(aucs)
        if isPrint == True:
            self.logger.info(
                "global_valid: True, epoch: {},  global_loss: {}, global_accuracy: {},  global_disparity:{}, global_pred_disparity: {},".format(
                    self.global_epoch,
                    global_loss,
                    global_auc,
                    global_dis,
                    global_pred_dis,
                )
            )

        return global_loss, global_acc, global_dis, global_pred_dis, global_auc

    def soften_losses(self, losses, delta):
        losses_list = torch.stack(losses)
        # loss = torch.max(torch.abs(losses_list))
        loss = torch.max(losses_list)

        alphas = F.softmax((losses_list - loss) / delta)
        alpha_without_grad = Variable(alphas.data.clone(), requires_grad=False)
        return alpha_without_grad, loss

    def train_Fed(self, epoch):
        self.model.train()
        self.optim.zero_grad()
        client_losses = []

        clients_weights = []
        self.weights_before = copy.deepcopy(self.model.get_params())

        for client_idx in range(self.n_clients):
            grads = {}
            self.model.set_params(self.weights_before)

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
            client_losses.append(loss)

            loss.backward()
            self.optim.step()

            new_weights = self.model.get_params()
            clients_weights.append(copy.deepcopy(new_weights))

        # for i in range(len(Deltas[0])):
        # add attack
        for j in range(0, len(clients_weights)):
            if self.attack_type == "Random":
                if self.random_list[j] < self.attack_ratio:
                    for key in self.weights_before:
                        clients_weights[j][key] = torch.rand_like(
                            clients_weights[j][key]
                        )
            if self.attack_type == "Enlarge":
                if self.random_list[j] < self.attack_ratio:
                    for key in self.weights_before:
                        clients_weights[j][key] = 2 * clients_weights[j][key]
            if self.attack_type == "Zero":
                if self.random_list[j] < self.attack_ratio:
                    for key in self.weights_before:
                        clients_weights[j][key] = 0.001 * clients_weights[j][key]

        new_solutions = {}
        for key in self.weights_before:
            tmp = clients_weights[0][key]
            for j in range(1, len(clients_weights)):
                tmp += clients_weights[j][key]
            new_solutions[key] = tmp / self.n_clients

        self.model.set_params(new_solutions)

        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            self.model.eval()
            losses, accs, client_disparities, pred_dis, aucs = self.local_valid_stage()
            losses, accs, client_disparities, pred_dis, aucs = self.global_valid_stage()
        self.global_epoch += 1

    def train(self):
        self.random_list = [random.uniform(0, 1) for i in range(self.n_clients)]

        start_epoch = self.global_epoch
        for epoch in range(start_epoch, self.max_epoch[0]):
            self.train_Fed(epoch)
