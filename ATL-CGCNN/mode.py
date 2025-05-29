import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len,
                                 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea, nbr_fea
        ], dim=2)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, 2 * self.atom_fea_len)).view(N, M, 2 * self.atom_fea_len)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network for property prediction
    (supports transfer learning via `freeze_conv`)
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, freeze_conv=False):
        """
        Parameters
        ----------
        orig_atom_fea_len : int
            Number of input atom features
        nbr_fea_len : int
            Number of input bond features
        atom_fea_len : int
            Hidden size in conv layers
        n_conv : int
            Number of conv layers
        h_fea_len : int
            Hidden size in FC layers
        n_h : int
            Number of FC layers
        classification : bool
            If True, outputs classification logits
        freeze_conv : bool
            If True, freezes conv layers for transfer learning
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.freeze_conv = freeze_conv

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        if self.freeze_conv:
            for param in self.convs.parameters():
                param.requires_grad = False

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)

        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, act in zip(self.fcs, self.softpluses):
                crys_fea = act(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(i) for i in crystal_atom_idx]) == atom_fea.shape[0]
        pooled = [torch.mean(atom_fea[idx], dim=0, keepdim=True) for idx in crystal_atom_idx]
        return torch.cat(pooled, dim=0)

    def get_trainable_parameters(self):
        """
        Returns only parameters that require gradients (for transfer learning optimizer)
        """
        return filter(lambda p: p.requires_grad, self.parameters())

