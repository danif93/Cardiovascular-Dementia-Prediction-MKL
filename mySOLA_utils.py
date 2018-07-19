"""
# MY SOLA NKW

class myMKL_srola:

    def __init__(self, X_list, K_type_list, y):

        # X_list: datasets list
        # K_name_list: names of kernels to use
        # y: ideal output vector

        self.y = y #used later in learning
        self.IK = np.dot(y, y.T) # used later in learning
        self.error = -1 #used later in learning
        self.Xtr_list = X_list
        self.num_datasets = len(X_list)
        self.num_K_types = len(K_type_list)
        self.num_samples = self.X_list[0].shape[0]

        self.eta = np.random.rand(self.num_datasets)
        self.lamb = np.random.rand(self.num_datasets, self.num_K_types)
        self.mu_list = [] # list of lists of matrices. every matrix refers to a kernel and to all its detasets


         #---------------------------------------------------------------------------------------------
         # get kernel objects and kernel matrices
        self.K_objects_list = [] # list of lists of objects that create the kernel matrices. first list = kernel objets list of first detaset
        self.K_list = [] # list of lists of kernels. first list associated to first dataset

        for X in X_list:
            dataset_kernel_objects_list = []
            dataset_kernel_list = []
            for K_type in K_type_list:

                k = kernel(X, self.num_K_type)  #small k object, big k matrix
                dataset_kernel_objects_list.append(k)
                dataset_kernel_list.append(k.kernelMatrix(X))

            self.K_objects_list.append(dataset_kernel_objects_list)
            self.K_list.append(dataset_kernel_list)
          #---------------------------------------------------------------------------------------------
          # randomly initialize mu vectors. every vector length depends on the kernel features. then we have a matrix of mu per kernel type
        for K in self.K_list[0]:
            self.mu_list.append(np.ones(self.num_datasets, K.shape[1]))


    def learning(self, tol = 0.01):

        while(True):
            self.learnMu()
            self.learnK()
            self.learnLambda()
            self.learnEta()
            error = self.computeError()
            if self.error < 0:
                continue
            if np.abs(error-self.error) < tol:
                break


    def getParam(self):

        return self.K_objects_list, self.mu_list, self.lamb, self.eta


    def test(self, X_list, goal = 'classification'):
        #X_list = test set

        K_test_list = []

        for dataset_index, X in enumerate(X_list):
            K_test_dataset_list = []
            for k_index, mu_k in enumerate(self.mu_list):
                tmp_X = X[:, np.where(mu_k[dataset_index] != 0)]
                K_test_dataset_list.append(self.k_objects_list[i][k_index].kernelMatrix(tmp_X))

            K_test_list.append(K_test_dataset_list)


        approximation = self.computeApproximation(K_test_list)
        if goal == 'classification':
            self.y_pred = classify(approximation, self.y)
        else:
            self.y_pred = regression(approximation)


    def computeApproximation(self, kernel_list):

        approximation = 0

        for dataset, K_list in enumerate(kernel_list):
            vec = np.zeros((self.num_samples))
            for kernel, K in enumerate(K_list):
                  vec += self.lamb[dataset, kernel] * K


            dataset_vec.append(vec)
        for i, d in enumerate(dataset_vec):
            approximation += self.eta[i] * d


        return approximation


    def learnMu(self):

        for k_index, mu_k in enumerate(self.mu_list):
            for d_index, mu in enumerate(mu_k):
                X = self.Xtr_list[d_index]
                alphas = np.arange(0.001, 3, 0.007)
                cl = LassoCV(alphas = alphas)
                mu = cl.fit(X, self.y).coef_
                mu = np.where(mu == 0, 0, 1)
                k = self.k_objects_list[d_index][k_index]
                k = kernel(X[:, np.where(mu != 0)], k.getType(), k.getParam())
                self.k_objects_list[d_index][k_index] = k
                self.K_list[d_index][k_index] = k.kernelMatrix(k.getMatrix())


    def learnK(self):

        apprximation = self.computeApproximation(self.K_list)
        actualError = frobeniusInnerProduct(approximation - self.IK, approximation - self.IK)
        tmp_K_list = self.K_list

        for i, k_list in enumerate(self.K_objects_list):
            for j, k in enumerate(k_list):
                if k.getType == 'linear':
                    continue
                interval = getParamInterval(k.getParam, k.getType)
                score = []
                matrices = []
                for param in interval:
                    matrices.append(k.setParam(param).kernelMatrix(k.getMatrix()))
                    tmp_K_list[i][j] = matrices[-1]
                    tmp_approximation = self.computeApproximation(tmp_K_list)
                    score.append(frobeniusInnerProduct(tmp_approximation - self.IK, tmp_approximation - self.IK))

                if actualError > np.min(score):
                    actualError = np.min(score)
                    tmp_K_list[i][j] = matrices[np.argmin(score)]
                else:
                    tmp_K_list[i][j] = self.K_list[i][j]

        self.K_list = tmp_K_list


    def learnLambda(self):

        for dataset_index, lamb in enumerate(self.lamb):

            C = np.zeros((self.K_list[0][0].shape[0], self.K_list[0][0].shape[0], len(self.K_list[0])))
            eta_ = 1/self.eta[dataset_index]
            for i, Ki in enumerate(self.K_list[dataset_index]):
                C[:,:,i] = Ki*eta_



            for dataset_index, K_list_dataset in enumerate(self.K_list):
                Di = np.zeros((self.num_samples, self.num_samples))
                for kernel, K in enumerate(K_list_dataset):
                    Di += K * self.lamb[dataset, kernel]

                C[:,:,i] = Di

            fixed_approximation = np.zeros(self.IK.shape)
            for d_index, K_list_dataset in enumerate(self.K_list):
                if d_index == dataset_index:
                    continue

                tmp_approximation = np.zeros(self.IK.shape)
                for kernel_index, Ke in enumerate(K_list_dataset):
                    tmp_approximation += Ke * self.lamb[d_index, kernel_index]

                fixed_approximation += self.eta[d_index] * tmp_approximation

            Y = self.IK - fixed_approximation

            A = np.zeros((len(self.K_list[0]), len(self.K_list[0])))
            B = np.zeros((len(self.K_list[0])))
            for row in C:
                for elem in row:
                   B += elem
                   A += np.dot(elem, elem.T)

            A += np.identity(A.shape[0]) #np.identity(...) * gamma_lambda for cross validation
            self.lamb[dataset_index, :] = np.dot(np.linalg.inv(A), np.dot(B, Y)) # TODO control dimensions
"""
"""
            K_ = [] #  K'
            for Ki in K:
                K_.append(Ki+ np.identity(Ki.shape[0])) #np.identity(...) * gamma_lambda for cross validation

            KTK = np.zeros((len(K_), len(K_), K_[0].shape[0], K_[0].shape[1]) # K'_ij
            for i, Ki in enumerate(K_):
                for j, Kj in enumerate(K_):
                    KTK[i, j, :, :] = np.dot(Ki, Kj)


            fixed_approximation = np.zeros(self.IK.shape)
            for d_index, K_list_dataset in enumerate(self.K_list):
                if d_index == dataset_index:
                    continue

                tmp_approximation = np.zeros(self.IK.shape)
                for kernel_index, Ke in enumerate(K_list_dataset):
                    tmp_approximation += Ke * self.lamb[d_index, kernel_index]

                fixed_approximation += self.eta[d_index] * tmp_approximation

            B = self.IK - fixed_approximation


            # HARD CODED

            K11_inv = np.linalg.inv(KTK[0,0,:,:])
            K21_K11 = np.dot(KTK[1,0,:,:], K11_inv)
            X = np.dot(K[1], self.IK) - np.dot(K21_K11, np.dot(K[0], self.IK))
            Y = KTK[1,2,:,:] - np.dot(K21_K11, KTK[0,2,:,:])
            Z = KTK[1,1,:,:] - np.dot(K21_K11, KTK[0,1,:,:])
            Z_inv = np.linalg.inv(Z)
            P = np.dot(K11_inv, (np.dot(K[0], self.IK) - np.dot(KTK[0,1,:,:], np.dot(X, Z_inv))))
            Q = np.dot(K11_inv, (np.dot(KTK[0,1,:,:], np.dot(Y, Z_inv)) - KTK[0,2,:,:]))
            M = np.dot(KTK[2,0,:,:], Q) - np.dot(KTK[2,1,:,:], np.dot(Y, Z_inv)) + KTK[2,2,:,:]
            M_inv = np.linalg.inv(M)
            N = np.dot(K[2], self.IK) - np.dot(KTK[2,0,:,:], P) - np.dot(KTK[2,1,:,:], np.dot(X, Z_inv))

            M_inv_N = np.dot(M_inv, N)
            self.lamb[dataset_index, 2] = M_inv_N[0,0]
            self.lamb[dataset_index, 0] = (P + np.dot(Q, M_inv_N))[0,0]
            self.lamb[dataset_index, 1] = np.dot((X - np.dot(Y, M_inv_N))), Z_inv)[0,0] """
"""

    def learnEta(self):

        C = np.zeros((self.K_list[0][0].shape[0], self.K_list[0][0].shape[0], len(self.Xtr_list)))
        for dataset_index, K_list_dataset in enumerate(self.K_list):
            Di = np.zeros((self.num_samples, self.num_samples))
            for kernel, K in enumerate(K_list_dataset):
                Di += K * self.lamb[dataset, kernel]

            C[:,:,i] = Di

        Y = self.IK.ravel()
        A = np.zeros((len(self.Xtr_list), len(self.Xtr_list)))
        B = np.zeros((len(self.Xtr_list)))
        for row in C:
            for elem in row:
                B += elem
                A += np.dot(elem, elem.T)

        A += np.identity(A.shape[0]) #np.identity(...) * gamma_eta for cross validation
        self.eta = np.dot(np.linalg.inv(A), np.dot(B, Y)) # TODO control dimensions

"""
"""
        D_ = [] #  D'
        for Di in D:
            D_.append(Di+ np.identity(Di.shape[0])) #np.identity(...) * gamma_eta for cross validation

        DTD = np.zeros((len(D_), len(D_), D_[0].shape[0], D_[0].shape[1]) # D'_ij
        for i, Di in enumerate(D_):
            for j, Dj in enumerate(D_):
                DTD[i, j, :, :] = np.dot(Di, Dj)

        # HARD CODED

        D11_inv = np.linalg.inv(DTD[0,0,:,:])
        D21_D11 = np.dot(DTD[1,0,:,:], D11_inv)
        X = np.dot(D[1], self.IK) - np.dot(D21_D11, np.dot(D[0], self.IK))
        Y = DTD[1,2,:,:] - np.dot(D21_D11, DTD[0,2,:,:])
        Z = DTD[1,1,:,:] - np.dot(D21_D11, DTD[0,1,:,:])
        Z_inv = np.linalg.inv(Z)
        P = np.dot(D11_inv, (np.dot(D[0], self.IK) - np.dot(DTD[0,1,:,:], np.dot(X, Z_inv))))
        Q = np.dot(D11_inv, (np.dot(DTD[0,1,:,:], np.dot(Y, Z_inv)) - DTD[0,2,:,:]))
        M = np.dot(DTD[2,0,:,:], Q) - np.dot(DTD[2,1,:,:], np.dot(Y, Z_inv)) + DTD[2,2,:,:]
        M_inv = np.linalg.inv(M)
        N = np.dot(D[2], self.IK) - np.dot(DTD[2,0,:,:], P) - np.dot(DTD[2,1,:,:], np.dot(X, Z_inv))

        M_inv_N = np.dot(M_inv, N)
        self.eta[2] = M_inv_N[0,0]
        self.eta[0] = (P + np.dot(Q, M_inv_N))[0,0]
        self.eta[1] = np.dot((X - np.dot(Y, M_inv_N))), Z_inv)[0,0]"""
"""