import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from scipy.linalg import lstsq
from scipy.sparse import csr_matrix

import math
import sys

# total arguments
n = len(sys.argv)
if n<3:
    sys.exit("\nArguments reuired in this order: address to the vector file, number of modes to eliminate, time-step,"
     "flag to calculate solution modes and time-dynamics (True or False).\n")

print("\nTotal arguments passed:", n)
sFileName = sys.argv[1]
numModes = int(sys.argv[2])
timeStep = float(sys.argv[3])
calModes = bool(sys.argv[4])

# Defining two functions that can read two different types of PETSc Mat objects
# Function to read PETSC_VIEWER_ASCII_PYTHON output of a PETSc Mat
def PETScRead_mpiaij(file):
    lines = file.read().splitlines()
    assert 'Mat Object:' in lines[0]
    assert lines[1] == '  type: seqaij'
    for line in lines[2:]:
        parts = line.split(': ')
        assert len(parts) == 2
        assert parts[0].startswith('row ')

        row_index = int(parts[0][4:])
        row_contents = eval(parts[1].replace(')  (', '), ('))

        # Here you have the row_index and a tuple of (column_index, value)
        # pairs that specify the non-zero contents. You could process this
        # depending on your needs, e.g. store the values in an array.
        for (col_index, value) in row_contents:
            #print('row %d, col %d: %s' % (row_index, col_index, value))
            # TODO: Implement real code here.
            # You probably want to do something like:
            data[row_index][col_index] = value

# Function to read MATLAB sparse matrix format
def ReadMatlabMat(sFileName):
    data = pd.read_csv(sFileName, sep=' ', skiprows=6, skipfooter=2, names=['i', 'j', 'remove_this', 'value'], engine='python')
    data.drop(columns= ['remove_this'], inplace= True)
    # renumbering array indices to make MATLAB format compatible to Python's
    data['i'] += -1
    data['j'] += -1
    # get the number of rows
    num_rows = max(data['i']) + 1
    num_cols = max(data['j']) + 1
    # Creating a sparse matrix
    A = csr_matrix((data['value'], (data['i'], data['j'])), shape= (num_rows, num_cols))
    return A

start = time.time()

print("reading data from file...")
data = ReadMatlabMat(sFileName)
X = data.todense() # converting a sparse matrix into a dense one
r = numModes # rank of the low-rank SVD approximation
dt = timeStep

print("Preparing Data...")
num_cols = X.shape[1]
end_time = dt * num_cols
t = np.arange(0, end_time, dt)

# Spliting the snapshots dataset into two slices - by chopping off first or last column
X1 = X[:, 0:-1]
X2 = X[:, 1:]

print('Starting DMD analysis.')
# Calculating rank-r truncation, using the first bunch in the data matrix
# Vstar is the conjugate transpose of V
print("Computing the SVD...")
U, S, Vstar = np.linalg.svd(X1, full_matrices=False)
V = Vstar.T.conj()
S_matrix = np.diag(S)
Ur = U[:, :r]
Sr = S_matrix[:r, :r]
Vr = V[:, :r]

# Best-fit regression of A
print('Solving LSQRT')
Atilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)

# ---------------------- Matrix Transformation -------------
I = np.identity(r)
Gtilde = np.linalg.inv(I - Atilde) @ Atilde
print(f"Condition number of Atilde: {np.linalg.cond(Atilde)}")
print(f"Condition number of (I - Atilde): {np.linalg.cond(I - Atilde)}")
update = Ur @ Gtilde @ (Ur.T @ X2[:, -1])

sOutputName = 'DMDUpdate.dat'
np.savetxt(sOutputName, update)
print(f"DMD over-relaxation update created succesfully, and saved as {sOutputName}")

if calModes:
    print("Calculating solution mode time-dynamics...")
    eigs, W = np.linalg.eig(Atilde)
    # Check the size of the eigenvectors
    assert W.shape == (r, r), "Size of eigenvectors is incorrect"

    # Reconstructing the high-dimensional DMD modes from the r sub-space
    print('Computing spatial modes')
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W
    sOutputName = 'SolnModesMat.dat'
    np.savetxt(sOutputName, Phi)

    omega = np.log(eigs)/dt
    assert np.all(np.isinf(omega)) == False, "Omega values have inf"

    # Doing least-squares to find initial solution
    x1 = X[:, 0]
    b, res, rnk, singularvalues = lstsq(Phi, x1)
    b = b.T
    time_dynamics = np.empty((len(t), r))
    sOutputName = 'time_dynamics.dat'
    np.savetxt(sOutputName, time_dynamics)

    for iter in range(0, len(t)):
        time_dynamics[iter, :] = b*np.exp(omega*t[iter])
        

    # You can play with this variable (time_multiplier) to predict further in the future
    time_multiplier = 20
    t2 = np.arange(0, time_multiplier*end_time, dt)
    time_dynamics2 = np.empty((len(t2), r))
    for iter in range(0, len(t2)):
        time_dynamics2[iter, :] = b*np.exp(omega*t2[iter])
        
    print("Computing Solution states")
    X_dmd = Phi @ time_dynamics.T
    X_dmd2 = Phi @ time_dynamics2.T
    sOutputName = 'time_dynamics2.dat'
    np.savetxt(sOutputName, time_dynamics2)

#-------------- error metrics------------
print("Calculating norms...")
dotRes = np.dot(Phi.T, Phi)
np.fill_diagonal(dotRes, 0)
modesNorm = np.linalg.norm(dotRes, 'fro')
print(f"\nFrobenius norm of DMD modes: {modesNorm}")

print(f"Total time: {time.time()-start} s")

print("------*** DMD out!! ***-------")
