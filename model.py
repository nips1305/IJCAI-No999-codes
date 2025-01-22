from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from core_qnn.quaternion_ops import *
import math
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric



def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)

def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian

def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree



class QGNNLayer(Module):
    def __init__(self, in_features, out_features, quaternion_ff=True,
                 act=F.relu, init_criterion='he', weight_init='quaternion',
                 seed=None):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.quaternion_ff = quaternion_ff
        self.act = act

        if self.quaternion_ff:
            self.register_parameter('r', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('i', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('j', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('k', Parameter(torch.Tensor(self.in_features, self.out_features)))
        else:
            self.register_parameter('commonLinear', Parameter(torch.Tensor(self.in_features, self.out_features)))

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters() # one time with hamilton matrix

    def reset_parameters(self):
        if self.quaternion_ff:
            winit = {'quaternion': quaternion_init,
                     'unitary': unitary_init}[self.weight_init]
            affect_init(self.r, self.i, self.j, self.k, winit,
                        self.rng, self.init_criterion)

        else:
            stdv = math.sqrt(6.0 / (self.commonLinear.size(0) + self.commonLinear.size(1)))
            self.commonLinear.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if x.device != self.r.device:
            x = x.to(self.r.device)
        if adj.device != self.r.device:
            adj = adj.to(self.r.device)
            
        if self.quaternion_ff:
            # Regular matrix multiplication
            r1 = torch.cat([self.r, -self.i, -self.j, -self.k], dim=0)
            i1 = torch.cat([self.i, self.r, -self.k, self.j], dim=0)
            j1 = torch.cat([self.j, self.k, self.r, -self.i], dim=0)
            k1 = torch.cat([self.k, -self.j, self.i, self.r], dim=0)
            self.hamilton_matrix = torch.cat([r1, i1, j1, k1], dim=1)
            out = torch.mm(adj, torch.mm(x, self.hamilton_matrix))
        else:
            out = torch.mm(adj, torch.mm(x, self.commonLinear))

        return self.act(out)


class HyReaL(Module):
    def __init__(self,
                 name,
                 X,
                 A,
                 labels,
                 layers=None,
                 acts=None,
                 max_epoch=10,
                 max_iter=50,
                 learning_rate=10 ** -2,
                 coeff_reg=10 ** -3,
                 seed=114514,
                 lam=-1,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                 ):
        super(HyReaL, self).__init__()
        self.name = name
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)

        self.n_clusters = self.labels.unique().shape[0]
        if layers is None:
            layers = [32, 16]
        self.layers = layers
        self.acts = acts
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        self.seed = seed

        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]

        self.indicator = self.X
        self.embedding = self.X
        self.links = 0
        self.lam = lam
        self._build_up()

    def _build_up(self):
        self.linear = torch.nn.Linear(self.input_dim, self.layers[0] * 4).to(self.device)
        self.quaternion_module_list = nn.ModuleList()
        
        for i in range(len(self.layers) - 1):
            self.quaternion_module_list.append(QGNNLayer(self.layers[i] * 4, self.layers[i + 1] * 4, quaternion_ff=True, act=self.acts[i], init_criterion='he', weight_init='quaternion', seed=self.seed))

    def forward(self, Laplacian):
        input = self.linear(self.X)
        
        # GNN layers
        for i in range(len(self.quaternion_module_list)):
            input = self.quaternion_module_list[i](input, Laplacian)
        
        # reshape and mean
        self.embedding = input.view(self.data_size, 4, self.layers[-1]).mean(dim=1)

        # reconstruct the adjacency matrix 
        recons_A = torch.matmul(self.embedding, self.embedding.t())

        return recons_A

    def _compute_regularization_loss(self):
        """L1 regularization for weights"""
        l1_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name or any(x in name for x in ['r', 'i', 'j', 'k']):
                l1_loss += torch.abs(param).sum()
        return l1_loss
    
    def _compute_structural_loss(self, recons_A):
        """Structural consistency loss"""
        size = self.X.shape[0]
        degree = recons_A.sum(dim=1)
        laplacian = torch.diag(degree) - recons_A
        
        structural_loss = torch.trace(
            self.embedding.t() @ laplacian @ self.embedding
        ) / size
        return structural_loss
    
    def _compute_reconstruction_loss(self, recons_A):
        """Calculate reconstruction loss"""
        epsilon = torch.tensor(10 ** -7).to(self.device)
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss_1 = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) \
                 + (1 - self.adjacency).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        return loss_1.sum() / (self.data_size ** 2)
    
    def _build_loss(self, recons_A):
        """Combined loss function with weighted components"""
        # Compute individual losses
        recon_loss = self._compute_reconstruction_loss(recons_A)
        reg_loss = self._compute_regularization_loss()
        struct_loss = self._compute_structural_loss(recons_A)
        
        # Combine losses with weights
        total_loss = recon_loss + \
                    self.coeff_reg * reg_loss + \
                    self.lam * struct_loss
        
        # Store individual losses for monitoring
        self.loss_components = {
            'reconstruction': recon_loss.item(),
            'regularization': reg_loss.item(),
            'structural': struct_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss

    def update_graph(self, embedding):
        weights = embedding.matmul(embedding.t())
        weights = weights.detach()
        return weights

    def clustering(self, weights):
        try:
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                weights = torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6)
            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree
            try:
                eigenvalues, vectors = torch.linalg.eigh(L)
            except RuntimeError:
                L_cpu = L.cpu()
                eigenvalues, vectors = torch.linalg.eigh(L_cpu)
                vectors = vectors.to(self.device)
            indicator = vectors[:, -self.n_clusters:].detach()
            km = KMeans(n_clusters=self.n_clusters, 
                    init='k-means++',
                    n_init=20,
                    max_iter=1000).fit(indicator.cpu().numpy())
            labels = km.labels_
            acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), labels)
            return acc, nmi, ari, f1
            
        except Exception as e:
            print(f"Clustering failed with error: {str(e)}")
            # Return worst case metrics
            return 0.0, 0.0, 0.0, 0.0        


    def run(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        weights = self.embedding.matmul(self.embedding.t()).detach()
        laplacian_weights = get_Laplacian_from_weights(weights)

        acc, nmi, ari, f1 = self.clustering(laplacian_weights)

        best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1

        print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        laplacian_adj = get_Laplacian(self.adjacency)

        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons_A = self(laplacian_adj)
                loss = self._build_loss(recons_A)
                loss.backward()
                optimizer.step()
                objs.append(loss.item())

            weights = self.embedding.matmul(self.embedding.t()).detach()
            laplacian_weights = get_Laplacian_from_weights(weights)

            acc, nmi, ari, f1 = self.clustering(laplacian_weights)
            loss = self._build_loss(recons_A)
            objs.append(loss.item())
            print('{}'.format(epoch) + 'loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (
            loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))

            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

        print("best_acc{},best_nmi{},best_ari{},best_f1{}".format(best_acc, best_nmi, best_ari, best_f1))
        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        print(acc_list.mean(), "±", acc_list.std())
        print(nmi_list.mean(), "±", nmi_list.std())
        print(ari_list.mean(), "±", ari_list.std())
        print(f1_list.mean(), "±", f1_list.std())
        return best_acc, best_nmi, best_ari, best_f1

    def build_pretrain_loss(self, recons_A):
        # Move epsilon tensor to the correct device
        epsilon = torch.tensor(10 ** -7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        
        # Ensure all operations are on the same device
        adjacency_sum = self.adjacency.sum()
        pos_weight = (self.data_size * self.data_size - adjacency_sum) / adjacency_sum
        
        # Ensure max operations are on the same device
        recons_A = recons_A.to(self.device)
        epsilon = epsilon.to(self.device)
        
        loss = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + \
               (1 - self.adjacency).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        
        loss = loss.sum() / (loss.shape[0] * loss.shape[1])
        loss_reg = self._compute_regularization_loss()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Ensure Laplacian is on the correct device
        Laplacian = get_Laplacian(self.adjacency).to(self.device)
        
        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            # Ensure recons_A is on the correct device
            recons_A = recons_A.to(self.device)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
        print(loss.item())




