#include "fargo3d.h"

void _Init(int index) {
  
  int i,j,k;
  
  real *sigma  = Density->field_cpu;
  real *vphi   = Vx->field_cpu;
  real *vr     = Vy->field_cpu;
  real *cs     = Energy->field_cpu;

  real eta = 0.75*ASPECTRATIO*ASPECTRATIO;
  
  i = j = k = 0;
  
  for (j=0; j<Ny+2*NGHY; j++) {
    for (i=0; i<Nx+2*NGHX; i++) {
      
      real vkmed = sqrt(G*MSTAR/ymed(j));
      real vkmin = sqrt(G*MSTAR/ymin(j));

      if (index == 0) {
        sigma[l] = SIGMA0*pow(ymed(j)/R0,-0.5);
        vr[l]    = 2*EPSILON*TS/(pow(TS,2) + pow(1+EPSILON,2))*eta*vkmin;
        vphi[l]  = vkmed;
        vphi[l] -= (pow(TS,2) + 1 + EPSILON)/(pow(TS,2)+pow(1+EPSILON,2))*eta*vkmed;
        cs[l]    = vkmed*ASPECTRATIO;
      }
      if (index == 1) {
        sigma[l] = EPSILON*SIGMA0*pow(ymed(j)/R0,-0.5);
        vr[l]    = -2*TS/(pow(TS,2) + pow(1+EPSILON,2))*eta*vkmin;
        vphi[l]  = vkmed;
        vphi[l] -= (1 + EPSILON)/(pow(TS,2)+pow(1+EPSILON,2))*eta*vkmed;
        cs[l]    = 0.0; // Pressureless fluid
      }
      vphi[l] -= OMEGAFRAME*ymed(j);
    }
  } 
}

void CondInit() {
   Fluids[0] = CreateFluid("gas",GAS);
   Fluids[1] = CreateFluid("dust",DUST);
   
   SelectFluid(0);
   _Init(0);
   
   SelectFluid(1);
   _Init(1);
     
   ColRate(1.0/TS, 0, 1, YES); //Fill the collision matrix with the
                               //inverse of the Stokes Number
}
