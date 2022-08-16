import pickle


class PerformanceMetrics:
    def __init__(self):
        self.epoch_idxs = []
        self.complexities = []
        self.recons = []
        self.comm_accs = []
        self.weights = []
        self.embed_r2 = []
        self.tok_r2 = []
        self.top_eng_acc = []  # Each entry is a 2-tuple of no-snap, snap accuracy
        self.syn_eng_acc = []
        self.top_val_eng_acc = []
        self.syn_val_eng_acc = []

    def add_data(self, epoch_idx, complexity, recons_loss, comm_acc, kl_weight, tok_r2, embed_r2, top_eng_acc,
                 syn_eng_acc, top_val_eng_acc, syn_val_eng_acc):
        self.epoch_idxs.append(epoch_idx)
        self.complexities.append(complexity)
        self.recons.append(recons_loss)
        self.comm_accs.append(comm_acc)
        self.weights.append(kl_weight)
        self.tok_r2.append(tok_r2)
        self.embed_r2.append(embed_r2)
        self.top_eng_acc.append(top_eng_acc)
        self.syn_eng_acc.append(syn_eng_acc)
        self.top_val_eng_acc.append(top_val_eng_acc)
        self.syn_val_eng_acc.append(syn_val_eng_acc)

    def to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            loaded = pickle.load(file)
        return loaded
