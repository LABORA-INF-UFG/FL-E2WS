import numpy as np
import ot

class EMD_Compute:
    def __init__(self):
        self.suporte = np.arange(10)

    def calcular_histograma(self, v_labels):
        contagens = np.array([np.sum(v_labels == rotulo) for rotulo in self.suporte ])
        total = contagens.sum()
        if total == 0:
            return np.zeros_like(contagens, dtype=float)
        return contagens / total

    def definir_matriz_custo(self, tipo='numerico'):

        n = len(self.suporte )
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if tipo == 'numerico':
                    C[i, j] = abs(self.suporte [i] - self.suporte [j])
                else:  # caso categórico
                    C[i, j] = 0 if self.suporte [i] == self.suporte [j] else 1
        return C

    def calcular_emd_penalizado(self, v_labels, gamma=5.0):

        # 1. Distribuição empírica
        P = self.calcular_histograma(v_labels)

        # 2. Distribuição base (uniforme)
        n = len(self.suporte )
        U = np.ones(n) / n

        # 3. Matriz de custo
        C = self.definir_matriz_custo()

        # 4. Calcular EMD (custo mínimo de transporte)
        emd_value = ot.emd2(P, U, C)

        # 5. Penalidade: contar quantos bins estão vazios na distribuição empírica
        missing_bins = np.sum(P == 0)
        penalty_factor = 1 + gamma * (missing_bins / n)

        # 6. EMD final penalizado
        emd_final = emd_value * penalty_factor
        return emd_final


    def compute_value(self, v_labels):
        # print("teste")
        return self.calcular_emd_penalizado(v_labels)









