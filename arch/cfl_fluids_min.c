#include "fargo3d.h"

void CflFluidsMin() {
  int i;
  real step = 1e30;
  real min;

  for (i=0;i<NFLUIDS;i++) {
    if (step > Min[i])
      step = Min[i];
  }
  
#ifdef FLOAT
  MPI_Allreduce(&step, &StepTime, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&step, &StepTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
  if(StepTime <= SMALLTIME) {
    masterprint("Error!!!--> Null dt\n");
    TimeStep = 99999;               // distinct number so it won't overwrite a real output
    MULTIFLUID(WriteOutputs(ALL)); // same call main.c uses for regular outputs
    masterprint("Output Before Crash\n");
    MPI_Barrier(MPI_COMM_WORLD);
    prs_exit(1);
  }
}
