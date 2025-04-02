
#include "fargo3d.h"

void rescale () {
SIGMA0 *= MSTAR/(R0*R0);
NU *= sqrt(G*MSTAR*R0);
OMEGAFRAME *= sqrt(G*MSTAR/(R0*R0*R0));
DT *= sqrt(R0*R0*R0/G/MSTAR);
}
