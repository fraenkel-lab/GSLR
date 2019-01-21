# netReg - a network-regularized generalized regression model

# Arguments:
#   -h [ --help ]                        Print this help.
#   -d [ --design ] arg                  Filename of the design matrix X.
#   -u [ --gx ] arg                      Filename of the affinity matrix GX for X.
#   -r [ --reponse ] arg                 Filename of the reponse matrix Y.
#   -v [ --gy ] arg                      Filename of the affinity matrix GY for Y.
#   -l [ --lambda ] arg (=1)             LASSO penalization parameter.
#   -s [ --psi ] arg (=0)                Penalization parameter for affinity matrix GX.
#   -x [ --xi ] arg (=0)                 Penalization parameter for affinity matrix GY.
#   -m [ --maxit ] arg (=100000)         Maximum number of iterations of coordinate descent. You should choose a sufficiently large number.
#   -t [ --threshold ] arg (=0.0000001)  Convergence threshold for coordinate descent. Anything below 0.0001 should suffice.
#   -o [ --outfile ] arg                 Filename of the output file.

# Model selection:
#   --modelselection                     Use modelselection, i.e. estimation of optimal shrinkage parameters using crossvalition, before doing the estimation of coefficients.
#   -n [ --nfolds ] arg (=10)            The number of cross-validation folds. This can be maximal the number of rows of X/Y and minimal 3.
#   -e [ --epsilon ] arg (=0.001)        Convergence threshold for the BOBYQA algorithm, i.e. the stop criterion for the model selection.
#   -b [ --bobit ] arg (=1000)           Maximal number of iterations for the BOBYQA algorithm.


# X.tsv:  samples are rows, columns are variables
# Y.tsv:  samples are rows, columns are classes
# GX.tsv: graph over the variables in X -- square matrix.

netReg -d X.tsv -r Y.tsv -u GX.tsv -l 10 -x 1 --maxit 1000 --threshold 0.0001 -o outfile.tsv


