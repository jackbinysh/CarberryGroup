#include <string>
/* internal global variables */
extern int n,LL;
extern double *nx,*ny,*nz;
extern double *hx,*hy,*hz;
extern double K1mK2,K3mK2;
extern double K1,K3;
extern double q0;
extern double W;
extern std::string RunName;

// summary stats functions
void SplayTwistBendDensities(int j, double& splaysq, double &twist, double &bend, double &twistsq);
void writeStatistics(FILE* fileStats);  // output measurements
void CalculateHopfionDimensionsXAxis(double* radius,  double* leftedge, double* rightedge, double* height);


inline void cross(double ax,double ay,double az,double bx,double by,double bz,double &rx,double &ry,double &rz);
int i_vr(int l);
int j_vr(int l);
int k_vr(int l);
