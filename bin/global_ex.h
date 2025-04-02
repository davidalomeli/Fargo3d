 /* This file was created automatically during compilation
from global.h. Do not edit. See python script
"scripts/global.py" for details. */ 

//MPI global variables

extern int CPU_Rank;
extern int CPU_Number;
extern boolean CPU_Master;

//Global variables


extern boolean Resistivity_Profiles_Filled;
extern boolean VxIsResidual;
extern boolean LogGrid;
extern boolean GuidingCenter;
extern boolean Corotating;
extern boolean Restart;
extern boolean Restart_Full;
extern boolean Stockholm;
extern boolean Merge;
extern boolean Merge_All;
extern boolean MonitorIntegral;
extern boolean TimeInfo;
extern boolean EverythingOnCPU;
extern boolean ForwardOneStep;
extern boolean Vtk2dat;
extern boolean Dat2vtk;
extern boolean PostRestart;
extern boolean OnlyInit;
extern boolean EarlyOutputRename;
extern boolean RedefineOptions;
extern boolean DeviceFileSpecified;
extern boolean StretchOldOutput;
extern boolean ThereArePlanets;
extern boolean ThereIsACentralBinary;
extern real    PhysicalTimeInitial;
extern real    PhysicalTime;
extern real    XAxisRotationAngle;
extern char    NewOutputdir[1024];
extern char    VersionString[1024];
extern char    StickyOptions[1024];
extern char    BoundaryFile[4096];
extern char    CommandLine[1024];
extern char    FirstCommand[1024];
extern char    ArchFile[1024];
extern char    ParameterFile[1024];
extern char    DefaultOut[1024];
extern char    DeviceFile[1024];
extern char    CurrentWorkingDirectory[1024];
extern char    *InputFile;
extern char    *PlanetaryFile;
extern char    *HostsList;
extern int     TimeStep;
extern int     FullArrayComms;
extern int     ContourComms;
extern int     DeviceManualSelection;
extern int     ArrayNb;
extern int     StretchNumber;
extern int     BinaryStar1;
extern int     BinaryStar2;

extern real Domega;

extern real InnerBorder;
extern real OuterBorder;

extern TimeProcess t_speedup_cpu;
extern TimeProcess t_speedup_gpu;
extern int t_speedup_count;
extern real time_speedup_cpu;
extern real time_speedup_gpu;

extern PlanetarySystem *Sys;
extern Point DiskOnPrimaryAcceleration;
extern Point IndirectTerm;

extern real StepTime;

extern real localforce[12];
extern real globalforce[12];

// MESH DENSITY CONSTANTS
extern real Xmc0;
extern real Xmc1;
extern real Xmc2;
extern real Xmc3;
extern real Xmc4;
extern real X_mesh_I;
				   
//FIELDS VARIABLES

extern Grid Gridd;

extern Field *Vx;
extern Field *Vy;
extern Field *Vz;

extern Field *Vx_temp;
extern Field *Vy_temp;
extern Field *Vz_temp;

extern Field *Vx_half;
extern Field *Vy_half;
extern Field *Vz_half;

extern Field *Slope;

extern Field *Mpx;
extern Field *Mpy;
extern Field *Mpz;
extern Field *Mmx;
extern Field *Mmy;
extern Field *Mmz;

extern Field *Pot;

extern Field *DivRho;
extern Field *DensStar;
extern Field *Qs;

extern Field *Density;
extern Field *Energy;
extern Field *Pressure;

extern Field *Total_Density;

extern Field *QL;
extern Field *QR;
extern Field *LapPPA;

extern Field *Sdiffyczc;
extern Field *Sdiffyfzc;
extern Field *Sdiffyczf;
extern Field *Sdiffyfzf;


// Below: fields specific to FARGO algorithms
extern Field2D *VxMed;
extern Field2D *Vxhy;
extern Field2D *Vxhyr;
extern Field2D *Vxhz;
extern Field2D *Vxhzr;
extern Field2D *Reduction2D;

extern FieldInt2D *Nxhy;
extern FieldInt2D *Nxhz;
extern FieldInt2D *Nshift;

extern Field *PhiStarmin;
extern Field *UStarmin;

//MHD FIELDS
//#ifdef MHD
extern Field *Bx;
extern Field *By;
extern Field *Bz;

extern Field *B1_star;
extern Field *B2_star;

extern Field *V1_star;
extern Field *V2_star;

extern Field *Slope_b1;
extern Field *Slope_v1;
extern Field *Slope_b2;
extern Field *Slope_v2;

extern Field *Emfx;
extern Field *Emfy;
extern Field *Emfz;

extern Field *EmfxH;
extern Field *EmfyH;
extern Field *EmfzH;
extern Field *BxH;
extern Field *ByH;
extern Field *BzH;
extern Field *Jx;
extern Field *Jy;
extern Field *Jz;
extern Field *EtaOhm;
extern Field *EtaHall;
extern Field *EtaAD;

extern Field *Divergence;
//#endif

extern Field2D *Density0;
extern Field2D *Vx0;
extern Field2D *Vy0;
extern Field2D *Vz0;
extern Field2D *Energy0;

//Communications variables

extern int Ncpu_x;
extern int Ncpu_y;

extern Buffer Bfd;
extern Buffer Bfu;
extern Buffer Bfl;
extern Buffer Bfr;
extern Buffer Bfcdl;
extern Buffer Bfcdr;
extern Buffer Bfcul;
extern Buffer Bfcur;

//Useful numbers

//CPU GLOBAL LIGHT ARRAYS
extern real *Xmin;
extern real *Ymin;
extern real *Zmin;
extern real *Xmed;
extern real *Ymed;
extern real *Zmed;
extern real *InvDiffXmed;
extern real *Sxi;
extern real *Sxj;
extern real *Sxk;
extern real *Syj;
extern real *Syk;
extern real *Szj;
extern real *Szk;
extern real *InvVj;
extern real shift_buffer[MAX1D];

//GPU GLOBAL LIGHT ARRAYS
extern real *Alpha;
extern real *Alpha_d;
extern real *Xmin_d;
extern real *Ymin_d;
extern real *Zmin_d;
extern real *Sxi_d;
extern real *InvDiffXmed_d;
extern real *Sxj_d;
extern real *Sxk_d;
extern real *Syj_d;
extern real *Syk_d;
extern real *Szj_d;
extern real *Szk_d;
extern real *InvVj_d;
extern real shift_buffer_d[MAX1D];

//Grid variables

extern int Nx;
extern int Ny;
extern int Nz;
extern int J;
extern int K;
extern int Y0;
extern int Z0;
extern int Stride_cpu;
extern int Stride_gpu;
extern int Pitch_cpu;
extern int Pitch_gpu;
extern int Pitch_Int_gpu;
extern int Pitch2D;
extern int Stride;
extern int ix;
extern int iy;
extern int ycells;
extern int zcells;
extern int y0cell;
extern int z0cell;
extern real Mindx;
//For checknan
extern Field *ListOfGrids;

//Boundary variables

extern int Bounl;
extern int Bounr;
extern int Bounu;
extern int Bound;

//psys variables.
extern real Xplanet;
extern real Yplanet;
extern real Zplanet;
extern real VXplanet;
extern real VYplanet;
extern real VZplanet;
extern real MplanetVirtual;

extern MPI_Status fargostat;

extern real OMEGAFRAME0;

extern int Fscan;

extern long VtkPosition;

//Multifluid variables
extern int Timestepcount;
extern int Fluidtype;
extern int FluidIndex;
extern real Min[NFLUIDS];
extern Fluid *Fluids[NFLUIDS];

//Pointers to functions
//WARNING!!! FUNCTIONS' ARGUMENTS MUST NOT CONTAIN BLANK SPACES
extern void (*ComputePressureFieldIso)();
extern void (*ComputePressureFieldAd)();
extern void (*ComputePressureFieldPoly)();
extern void (*SubStep1_x)(real);
extern void (*SubStep1_y)(real);
extern void (*SubStep1_z)(real);
extern void (*SubStep2_a)(real);
extern void (*SubStep2_b)(real);
extern void (*SubStep3)(real);
extern void (*DivideByRho)(Field*);
extern void (*VanLeerX_a)(Field*);
extern void (*VanLeerX_b)(real,Field*,Field*,Field*);
extern void (*VanLeerY_a)(Field*);
extern void (*VanLeerY_b)(real,Field*,Field*);
extern void (*VanLeerZ_a)(Field*);
extern void (*VanLeerZ_b)(real,Field*,Field*);
extern void (*momenta_x)();
extern void (*momenta_y)();
extern void (*momenta_z)();
extern void (*reduction_SUM)(Field*,int,int,int,int);
extern void (*reduction_MIN)(Field*,int,int,int,int);
extern void (*UpdateX)(real,Field*,Field*,Field*);
extern void (*UpdateY)(real,Field*,Field*);
extern void (*UpdateZ)(real,Field*,Field*);
extern void (*UpdateDensityX)(real,Field*,Field*);
extern void (*UpdateDensityY)(real,Field*);
extern void (*UpdateDensityZ)(real,Field*);
extern void (*NewVelocity_x)();
extern void (*NewVelocity_y)();
extern void (*NewVelocity_z)();
extern void (*AdvectSHIFT)(Field*,FieldInt2D*);
extern void (*AdvectRAM)(real,Field*);
extern void (*AdvectRAMlin)(real,Field*);
extern void (*RamComputeUstar)(real);
extern void (*RamSlopes)(Field*);
extern void (*ComputeResidual)(real);
extern void (*ComputeVweight)(Field*,Field*);
extern void (*ChangeFrame)(int,Field*,Field2D*);
extern void (*Potential)();
extern void (*CorrectVtheta)(real);
extern void (*cfl)(void);
extern void (*_ComputeForce)(real,real,real,real,real);
extern void (*copy_velocities)(int);
extern void (*VanLeerX_PPA_a)(Field*);
extern void (*VanLeerX_PPA_b)(Field*);
extern void (*VanLeerX_PPA_steep)(Field*);
extern void (*VanLeerX_PPA_c)(Field*);
extern void (*VanLeerX_PPA_d)(real,Field*,Field*,Field*);
extern void (*VanLeerX_PPA_d_2d)(real,Field*,Field*,Field2D*);
extern void (*mon_dens)();
extern void (*mon_momx)();
extern void (*mon_momy)();
extern void (*mon_momz)();
extern void (*mon_torq)();
extern void (*mon_reynolds)();
extern void (*mon_maxwell)();
extern void (*mon_bxflux)();
extern void (*comm)(int);
extern void (*Reset_field)(Field*);
extern void (*ComputeTotalDensity)();
extern void (*copy_field)(Field*,Field*);
//DUST DIFFUSION
extern void (*DustDiffusion_Core)(real);
extern void (*DustDiffusion_Coefficients)();
//MHD..........................................
extern void (*ComputeSlopes)(int,int,int,Field*,Field*);
extern void (*_ComputeStar)(real,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*);
extern void (*_ComputeEmf)(real,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*,Field*);
extern void (*_UpdateMagneticField)(real,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
extern void (*_LorentzForce)(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field*,Field*);
extern void (*_Resist)(int,int,int,int,int,int,int,int,int,Field*,Field*,Field*,Field2D*);
extern void (*EMF_Upstream_Integrate)(real);

extern void (*ComputeJx)();
extern void (*ComputeJy)();
extern void (*ComputeJz)();

extern void (*_OhmicDiffusion_emf)(int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
extern void (*OhmicDiffusion_coeff)();
extern void (*HallEffect_emfx)();
extern void (*HallEffect_emfy)();
extern void (*HallEffect_emfz)();
extern void (*_HallEffect_UpdateB)(real,int,int,int,int,int,int,int,int,int,int,int,Field*,Field*,Field*);
extern void (*HallEffect_UpdateEmfs)();
extern void (*HallEffect_coeff)();
extern void (*AmbipolarDiffusion_emfx)();
extern void (*AmbipolarDiffusion_emfy)();
extern void (*AmbipolarDiffusion_emfz)();
extern void (*AmbipolarDiffusion_coeff)();
//.............................................

extern void (*StockholmBoundary)(real);
extern void (*visctensor_cart)();
extern void (*addviscosity_cart)(real);
extern void (*visctensor_cyl)();
extern void (*addviscosity_cyl)(real);
extern void (*visctensor_sph)();
extern void (*addviscosity_sph)(real);
extern void (*Fill_GhostsX)();
extern void (*CheckMuteY)();
extern void (*CheckMuteZ)();

extern void (*SetupHook1)();

extern void (*_collisions)(real,int,int,int,int);
extern void (*Floor)();

extern void (*__WriteField)(Field*, int);
extern void (*__Restart)(Field*,int);

extern void (*boundary_ymin[NFLUIDS])();
extern void (*boundary_ymax[NFLUIDS])();
extern void (*boundary_zmin[NFLUIDS])();
extern void (*boundary_zmax[NFLUIDS])();
