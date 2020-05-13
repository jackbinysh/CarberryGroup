#include "Constants.h"
#include <string>

/* config file*/
std::string knot_filename;
std::string output_dir;
int NumComponents;
int NumRefinements;
int Nx,Ny,Nz;
bool Scaling,ScaleProportionally;
double BoxFractionx, BoxFractiony, BoxFractionz;
double Threshold_ninf_dot_n;
double Initialninftyx,Initialninftyy,Initialninftyz;
std::vector<int> degrees(0);
double TubeRadius;
