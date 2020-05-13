/**************************************************************/
/*      Finite difference code for knotted Hopf Skyrmions     */
/*      created April 2019: Jack Binysh, Gareth Alexander     */
/**************************************************************/

#include "nematic.h"    
#include "SolidAngle.h"    
#include "Geometry.h"
#include "InputOutput.h"
#include "Constants.h"    
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
//#include <complex>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
void startconfig(void);
void update(void);

int n,LL;
double *nx,*ny,*nz;
double *hx,*hy,*hz;
double K1mK2,K3mK2;
double q0;
double W; 
std::string RunName;

//int main(int argc, char** argv) 
int main (int argc, char*argv[])
{
  InitialiseSystemParameters(argv);

  nx=new double[LL];
  ny=new double[LL];
  nz=new double[LL];
  hx=new double[LL];
  hy=new double[LL];
  hz=new double[LL];

  startconfig();

  // the summary stats file
  char buf[128];
  FILE *fileStats;
  sprintf(buf,"%s/%s_summary_statistics.dat",output_dir.c_str(),RunName.c_str());
  fileStats=fopen(buf,"a");
  setbuf(fileStats, NULL);
  //fprintf(filestats, "%s %e %e %e %e %e %e %e %e %e %e %e %e\n",std::to_string(n).c_str(),nematicfreeenergy,cholestericfreeenergy,nematicfreeenergy+cholestericfreeenergy,nematicfreeenergyxzplane,cholestericfreeenergyxzplane,twistxzplane,splaysqbendsqxzplane,twistsqxzplane,radius,leftedge,rightedge,height); 
  fprintf(fileStats, "timestep  NematicFreeEnergy CholestericFreeEnergy TotalFreeEnergy XZNematicFreeEnergy XZTwistFreeEnergy XZTwist XZSplayBend XZTwistsq Radius LeftEdge RightEdge Height\n"); 

  cout << "starting simulation" << endl;
#pragma omp parallel default(none) shared (n,fileStats)
  while(n<=Nmax)
  {
#pragma omp single
    {
      if (n%stepskip==0)
      {
        writeVTKFiles();  // output VTK files for use with ParaView 
      }

      if (n%stepskipstatistics==0)
      {
        writeStatistics(fileStats);  // output measurements
      }
      n++;
    }
    update();
  }
} // end main

/**********************************************************************/
void startconfig(void)
{
  switch(InitialisationMethod)
  {
    case FROM_SOLIDANGLE:
      {
        InitialiseSolidAngle();

        int j,k,l,m;
        double omega,alpha,R,rho;
        // set initialisation type

        // initialise the knot
        Link Curve;
        if(0==InitialiseFromFile(Curve))
        {
          cout << "Filling in the Geometry of the Input Curve \n";
          ComputeGeometry(Curve);
          OutputScaledKnot(Curve); 
        }

        k=l=m=0;  
        omega=0.0; // overly cautious!!

        int c; // need to adapt to number of components
        c=0; // no real reason for this    
        R = TubeRadius; // radius for the tubular neighbourhood of the curve 
        alpha=0.0; // overly cautious!!
        viewpoint Point;

        for (j=0; j<LL; j++) 
        {    
          double tempnx = 0.0;
          double tempny = 0.0;
          double tempnz = 1.0;

          Point.xcoord = x(k); //1.0*k-Nx/2.0+0.5;
          Point.ycoord = y(l); //1.0*l-Ny/2.0+0.5;
          Point.zcoord = z(m); //1.0*m-Nz/2.0+0.5;

          // calculate the distance to the curve 
          rho = ComputeDistanceOnePoint(Curve,Point);
          // put a Skyrmion texture inside the tubular neighbourhood
          if (rho < R)
          {
            // here is the solid angle
            omega = ComputeSolidAngleOnePoint(Curve,Point);
            // inelegant method for getting longitudinal phase and component
            alpha = ComputeLongitudinalPhase(Curve,Point);
            c = WhichComponent(Curve,Point);  
            // set the director  
            // Chirality +1 is right handed.
            // Chirality -1 is left handed
            tempnx = sin(M_PI*rho/R)*cos(0.5*omega-degrees[c]*alpha);
            tempny = sin(M_PI*rho/R)*sin(0.5*omega-degrees[c]*alpha);
            tempnz = -cos(M_PI*rho/R);
          }

#if BC // Dirichlet boundary conditions 
          if (m==0) 
          {
            tempnx = 0.0;
            tempny = 0.0;
            tempnz = 1.0;
          }
          if (m==Nz-1)
          { 
            tempnx = 0.0;
            tempny = 0.0;
            tempnz = 1.0;
          }
#endif
          nx[j]=tempnx;
          ny[j]=tempny;
          nz[j]=tempnz;
          k++;
          if (k==Nx) {l++; k=0;}
          if (l==Ny) {m++; l=0;}
        }
        break;
      }
    case FROM_FUNCTION:
      {

        int CHOLESTERIC,SKYRMION,SKYRMIONBAG,SQUARE,HEXAGONAL,HOPF,TORON,CONST;
        //define texture to simulate -- 0 or 1
        CONST = 0;
        CHOLESTERIC = 0;
        SKYRMION = 0;
        SKYRMIONBAG = 0;
        TORON = 1;
        HOPF = 0;

        if(CONST==1)
        {
          double k0,l0,m0;
          k0 = Nx/2.0 - 0.5; l0 = Ny/2.0 - 0.5; m0 = Nz/2.0 - 0.5; // offset from the lattice points
          for (int j=0; j<LL; j++) 
          {
            // whats our 3D position?
            int k = i_vr(j);
            int l = j_vr(j);
            int m = k_vr(j);
            if(k<k0)
            {
              nx[j]=0;
              ny[j]=0;
              nz[j]=1;
            }
            else
            {
              nx[j]=0;
              ny[j]=0;
              nz[j]=-1;
            }
          }
        }

        if(CHOLESTERIC==1)
        {
          for (int j=0; j<LL; j++) 
          {
            // whats our 3D position?
            int k = i_vr(j);
            int l = j_vr(j);
            int m = k_vr(j);
            //nx[j]=-sin(q0*m);
            //ny[j]=cos(q0*m);
            //nz[j]=0;

            //nx[j]=cos(theta);
            //ny[j]=sin(theta)*sin(tempq0*k*k);
            //nz[j]=sin(theta)*cos(tempq0*k*k);
            //

            // nx[j]=sin(theta)*cos(tempq0*l*l);
            // ny[j]=cos(theta);
            // nz[j]=sin(theta)*sin(tempq0*l*l);
            //
            double theta=0.1;
            nx[j]=-sin(theta)*sin(q0*m);
            ny[j]=sin(theta)*cos(q0*m);
            nz[j]=cos(theta);
          }
        }

        if(SKYRMION==1)
        {
          double k0,l0,m0;
          k0 = Nx/2.0 - 0.5; l0 = Ny/2.0 - 0.5; m0 = Nz/2.0 - 0.5; // offset from the lattice points
          double r,R,phi;
          R = Nz/2;
          int j,k,l,m;
          j=k=l=m=0;
          for (j=0; j<LL; j++) 
          {
            // director field
           // nx[j] = cos(q0*m);
           // ny[j] = sin(q0*m);
           // nz[j] = 0;
            // replace a cylinder with a twisted Skyrmion -- testing!
            r = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            if (r<R)
            {
              phi = atan2(l-l0,k-k0);

              // nz is common to the twisted and untwisted cases
              nz[j] = cos(2*M_PI*r/R);

              // degree +1 , untwisted
              nx[j] = sin(2*M_PI*r/R)*sin(phi);
              ny[j] = -sin(2*M_PI*r/R)*cos(phi);
              // polarisation for +1 untwisted
              // degree -1
              //	nx[j] = -sin(M_PI*r/R)*sin(phi);
              //	ny[j] = -sin(M_PI*r/R)*cos(phi);
            }

#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif

            // deal with the periodic boundaries
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }
        }

        if(SKYRMIONBAG==1)
        {
          double k0,l0,m0;
          k0 = Nx/2.0 - 0.5; l0 = Ny/2.0 - 0.5; m0 = Nz/2.0 - 0.5; // offset from the lattice points
          double r,R,phi;
          R = Nz/2.0;
          int j,k,l,m;
          j=k=l=m=0;
          for (j=0; j<LL; j++) 
          {
            nx[j] =0;
            ny[j] = 0;
            nz[j] = 1;

            r = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            if (r<R)
            {
              phi = atan2(l-l0,k-k0);
              // nz is common to the twisted and untwisted cases
              nz[j] = cos(2*M_PI*r/R);
              // degree +1 , untwisted
              nx[j] = sin(2*M_PI*r/R)*sin(phi);
              ny[j] = -sin(2*M_PI*r/R)*cos(phi);
            }

#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif

            // deal with the periodic boundaries
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }

          // now interpolate smoothly to a flat texture
          j=k=l=m=0;
          int upperlimit1=(Nz)/2;
          int upperlimit2=3*Nz/4;
          for (j=0; j<LL; j++) 
          {
            r = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            phi = atan2(l-l0,k-k0);

            double Dm=upperlimit2-upperlimit1;
            if(r<R && m>upperlimit1)
            {
              double deltam = m-upperlimit1;
              double t=(M_PI/2)*(deltam/Dm);
              // nz is common to the twisted and untwisted cases
              nz[j] = cos(2*M_PI*r/R);
              // degree +1 , untwisted
              nx[j] = sin(2*M_PI*r/R)*sin(phi+t);
              ny[j] = -sin(2*M_PI*r/R)*cos(phi+t);
            }

#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif
            // deal with the periodic boundaries
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }

          // now interpolate smoothly to a flat texture
          j=k=l=m=0;
          for (j=0; j<LL; j++) 
          {
            r = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            phi = atan2(l-l0,k-k0);

            double Dm=Nz-upperlimit2;
            if(r<R && m>upperlimit2)
            {
              double deltam = m-upperlimit2;
              double t=(M_PI)*(deltam/Dm);
              double sgn = (cos(2*M_PI*r/R)>0)-(cos(2*M_PI*r/R)<0);
             double q=(t/2)*(1-cos(2*M_PI*r/R));
              // nz is common to the twisted and untwisted cases
              // degree +1 , untwisted
              nx[j] = sin(2*M_PI*r/R)*sin(phi+(M_PI/2))+sin(q)*cos(2*M_PI*r/R)*sin(phi);
              ny[j] = -sin(2*M_PI*r/R)*cos(phi+(M_PI/2.0))-sin(q)*cos(2*M_PI*r/R)*cos(phi);
              nz[j] = cos(q)*cos(2*M_PI*r/R);
            }
#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif
            // deal with the periodic boundaries
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }
        }

        if (TORON == 1)
        {
          double k0,l0,m0,rr,rho,RR;    // variables for a toron 

          k0 = Nx/2.0 - 0.5; l0 = Ny/2.0 - 0.5; m0 = Nz/2.0 - 0.5; // offset from the lattice points
          RR = 32.0; q0 = M_PI/(2.0*RR); // scale

          int j,k,l,m;
          j=k=l=m=0;
          for (j=0; j<LL; j++)
          {
            // the following code should initialise a toron
            // not quite fully general yet
            nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;
            rr = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            rho = sqrt((rr-RR)*(rr-RR)+(m-m0)*(m-m0));

            if ((rr<=2.0*RR)&&(fabs(m-m0)<RR+3.0))
            {
              nx[j] = ((k-k0)/rr)/sqrt(2.0);
              ny[j] = ((l-l0)/rr)/sqrt(2.0);
              nz[j] = 1.0/sqrt(2.0);
              if (m>m0) {nx[j] = -nx[j]; ny[j] = -ny[j];} 
              if ((rr<RR)&&(fabs(m-m0)<RR)) {nz[j] = -nz[j];}
            }
            if (rho<RR)
            {
              nx[j] = -cos(q0*rho)*(l-l0)/rr - sin(q0*rho)*((m-m0)/rho)*((k-k0)/rr);
              ny[j] = cos(q0*rho)*(k-k0)/rr - sin(q0*rho)*((m-m0)/rho)*((l-l0)/rr);
              nz[j] = sin(q0*rho)*(rr-RR)/rho;
            }

#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif

            // deal with the periodic boundaries            
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }
        }

        if (HOPF == 1)
        {
          int xup,xdwn,yup,ydwn,zup,zdwn;
          double rho,theta,phi, rr, norm;    // variables for hopf texture
          double k0,l0,m0;   
          k0 = Nx/2.0 - 0.5; l0 = Ny/2.0 - 0.5; m0 = Nz/2.0 - 0.5; // offset from the lattice points

          // the radius of the central core
          double RR = 0.2*Nx;  // scale
          // how fat the hopfion should be
          double RR2 =0.45*Nz;

          int j,k,l,m;
          j=k=l=m=0;
          for (int j=0; j<LL; j++)
          {
            // director field
            nx[j] = 0;
            ny[j] = 0;
            nz[j] = 1;
            // back to Hopf
            rr = sqrt((k-k0)*(k-k0)+(l-l0)*(l-l0));
            rho = sqrt((rr-RR)*(rr-RR)+(m-m0)*(m-m0));
            if (rho < RR2)
            {
              theta = atan2(m-m0,rr-RR);
              if (theta < 0.0) {theta+=2.0*M_PI;} // put in the range [0,2pi)
              phi = atan2(l-l0,k-k0);
              //        phi = 0;
              if (phi < 0.0) {phi+=2.0*M_PI;} // put in the range [0,2pi)

              // cross-section looks like double twist and an escaped -1
              // for -theta these are ordered -1 inside and double twist outside; for +theta it is the other way around
              // Hopf invariant -1
              //nx[j] = -sin(M_PI*rho/RR2)*sin(phi-theta); // could also use -theta; they seem to be equivalent in energy
              //ny[j] = sin(M_PI*rho/RR2)*cos(phi-theta); // could also use -theta
              // Hopf invariant +1
              //nx[j] = -sin(0.5*M_PI*rho/RR2)*sin(phi+theta); // could also use -theta; they seem to be equivalent in energy
             // ny[j] = sin(0.5*M_PI*rho/RR2)*cos(phi+theta); // could also use -theta
             // nz[j] = -cos(0.5*M_PI*rho/RR2);
              // Hopf invariant 0 
              nx[j] = -sin(M_PI*rho/RR2)*sin(phi+theta); // could also use -theta; they seem to be equivalent in energy
              ny[j] = sin(M_PI*rho/RR2)*cos(phi+theta); // could also use -theta
              nz[j] = -cos(M_PI*rho/RR2);
            }

#if BC // normal anchoring along z
            if (m==0) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
            if (m==Nz-1) {nx[j] = 0.0; ny[j] = 0.0; nz[j] = 1.0;}
#endif
            // deal with the periodic boundaries
            k++;
            if (k==Nx) {l++; k=0;}
            if (l==Ny) {m++; l=0;}
          }
        }
      }
  } // end startconfig
}

/**********************************************************************/

void update(void)
{
  int j,k,l,m,jj;
  int xup,xdwn,yup,ydwn,zup,zdwn;
  int xyuu,xyud,xydu,xydd,xzuu,xzud,xzdu,xzdd,yzuu,yzud,yzdu,yzdd;
  double Dxnx,Dynx,Dznx,Dxxnx,Dyynx,Dzznx,Dxynx,Dxznx,Dyznx;
  double Dxny,Dyny,Dzny,Dxxny,Dyyny,Dzzny,Dxyny,Dxzny,Dyzny;
  double Dxnz,Dynz,Dznz,Dxxnz,Dyynz,Dzznz,Dxynz,Dxznz,Dyznz;
  // the mirror 2nd derivs
  double Dyxnx, Dzxnx, Dzynx, Dyxny, Dzxny, Dzyny, Dyxnz, Dzxnz, Dzynz;
  double hdotn,sqrtndotn;

  int Lx=Nx;
  int Ly=Ny;
  int Lz=Nz;

  k=l=m=0;

  /* Calculate derivatives, molecular field and the energy */
#pragma omp for
  for (j=0; j<LL; j++) 
  {
    // whats our 3D position?
    k = i_vr(j);
    l = j_vr(j);
    m = k_vr(j);

    // define neighbouring nodes
    xup=j+1; xdwn=j-1; yup=j+Lx; ydwn=j-Lx; zup=j+Lx*Ly; zdwn=j-Lx*Ly;
    xyuu=j+1+Lx; xyud=j+1-Lx; xydu=j-1+Lx; xydd=j-1-Lx;
    xzuu=j+1+Lx*Ly; xzud=j+1-Lx*Ly; xzdu=j-1+Lx*Ly; xzdd=j-1-Lx*Ly;
    yzuu=j+Lx+Lx*Ly; yzud=j+Lx-Lx*Ly; yzdu=j-Lx+Lx*Ly; yzdd=j-Lx-Lx*Ly;

    // correct for periodic boundaries
    if (k==0) {xdwn+=Lx; xydu+=Lx; xydd+=Lx; xzdu+=Lx; xzdd+=Lx;}
    if (k==Lx-1) {xup-=Lx; xyuu-=Lx; xyud-=Lx; xzuu-=Lx; xzud-=Lx;}
    if (l==0) {ydwn+=Lx*Ly; xyud+=Lx*Ly; xydd+=Lx*Ly; yzdu+=Lx*Ly; yzdd+=Lx*Ly;}
    if (l==Ly-1) {yup-=Lx*Ly; xyuu-=Lx*Ly; xydu-=Lx*Ly; yzuu-=Lx*Ly; yzud-=Lx*Ly;}
    if (m==0) {zdwn+=LL; xzud+=LL; xzdd+=LL; yzud+=LL; yzdd+=LL;}
    if (m==Lz-1) {zup-=LL; xzuu-=LL; xzdu-=LL; yzuu-=LL; yzdu-=LL;}

#if BC // cheap fix for the Dirichlet boundary conditions
    if (m==0) {xup=j; xdwn=j; yup=j; ydwn=j; zup=j; zdwn=j;}
    if (m==Lz-1) {xup=j; xdwn=j; yup=j; ydwn=j; zup=j; zdwn=j;}
#endif

    // calculate first order derivatives
    Dxnx = (nx[xup]-nx[xdwn])/2.0;
    Dynx = (nx[yup]-nx[ydwn])/2.0;
    Dznx = (nx[zup]-nx[zdwn])/2.0;

    Dxny = (ny[xup]-ny[xdwn])/2.0;
    Dyny = (ny[yup]-ny[ydwn])/2.0;
    Dzny = (ny[zup]-ny[zdwn])/2.0;

    Dxnz = (nz[xup]-nz[xdwn])/2.0;
    Dynz = (nz[yup]-nz[ydwn])/2.0;
    Dznz = (nz[zup]-nz[zdwn])/2.0;

    // calculate second order derivatives
    Dxxnx = nx[xup]-2.0*nx[j]+nx[xdwn];
    Dyynx = nx[yup]-2.0*nx[j]+nx[ydwn];
    Dzznx = nx[zup]-2.0*nx[j]+nx[zdwn];

    Dxxny = ny[xup]-2.0*ny[j]+ny[xdwn];
    Dyyny = ny[yup]-2.0*ny[j]+ny[ydwn];
    Dzzny = ny[zup]-2.0*ny[j]+ny[zdwn];

    Dxxnz = nz[xup]-2.0*nz[j]+nz[xdwn];
    Dyynz = nz[yup]-2.0*nz[j]+nz[ydwn];
    Dzznz = nz[zup]-2.0*nz[j]+nz[zdwn];

    // needed for unequal elastic constants
    Dxynx = 0.25*(nx[xyuu]-nx[xyud]-nx[xydu]+nx[xydd]);
    Dxznx = 0.25*(nx[xzuu]-nx[xzud]-nx[xzdu]+nx[xzdd]);
    Dyznx = 0.25*(nx[yzuu]-nx[yzud]-nx[yzdu]+nx[yzdd]);

    Dxyny = 0.25*(ny[xyuu]-ny[xyud]-ny[xydu]+ny[xydd]);
    Dxzny = 0.25*(ny[xzuu]-ny[xzud]-ny[xzdu]+ny[xzdd]);
    Dyzny = 0.25*(ny[yzuu]-ny[yzud]-ny[yzdu]+ny[yzdd]);

    Dxynz = 0.25*(nz[xyuu]-nz[xyud]-nz[xydu]+nz[xydd]);
    Dxznz = 0.25*(nz[xzuu]-nz[xzud]-nz[xzdu]+nz[xzdd]);
    Dyznz = 0.25*(nz[yzuu]-nz[yzud]-nz[yzdu]+nz[yzdd]);

    // set the mirror derivatives equal, for readability
    Dyxnx=Dxynx;
    Dzxnx=Dxznx;
    Dzynx=Dyznx;

    Dyxny=Dxyny;
    Dzxny=Dxzny;
    Dzyny=Dyzny;

    Dyxnz=Dxynz;
    Dzxnz=Dxznz;
    Dzynz=Dyznz;


    /* GARETHS ORIGINAL IMPLEMENTATION*/

    double thx,thy,thz;

    // calculate molecular field
    hx[j] = K2*(Dxxnx+Dyynx+Dzznx) - 2.0*K2*q0*(Dynz-Dzny);
    hy[j] = K2*(Dxxny+Dyyny+Dzzny) - 2.0*K2*q0*(Dznx-Dxnz);
    hz[j] = K2*(Dxxnz+Dyynz+Dzznz) - 2.0*K2*q0*(Dxny-Dynx);

    // unequal elastic constants
    hx[j] += K1mK2*(Dxxnx+Dxyny+Dxznz);
    hy[j] += K1mK2*(Dxynx+Dyyny+Dyznz);
    hz[j] += K1mK2*(Dxznx+Dyzny+Dzznz);

    hx[j] += K3mK2*(nx[j]*(nx[j]*Dxxnx+ny[j]*Dxynx+nz[j]*Dxznx)+ny[j]*(nx[j]*Dxynx+ny[j]*Dyynx+nz[j]*Dyznx)+nz[j]*(nx[j]*Dxznx+ny[j]*Dyznx+nz[j]*Dzznx)
        +(Dxnx+Dyny+Dznz)*(nx[j]*Dxnx+ny[j]*Dynx+nz[j]*Dznx)
        -(Dxny-Dynx)*(nx[j]*Dxny+ny[j]*Dyny+nz[j]*Dzny)-(Dxnz-Dznx)*(nx[j]*Dxnz+ny[j]*Dynz+nz[j]*Dznz));
    hy[j] += K3mK2*(nx[j]*(nx[j]*Dxxny+ny[j]*Dxyny+nz[j]*Dxzny)+ny[j]*(nx[j]*Dxyny+ny[j]*Dyyny+nz[j]*Dyzny)+nz[j]*(nx[j]*Dxzny+ny[j]*Dyzny+nz[j]*Dzzny)
        +(Dxnx+Dyny+Dznz)*(nx[j]*Dxny+ny[j]*Dyny+nz[j]*Dzny)
        -(Dynx-Dxny)*(nx[j]*Dxnx+ny[j]*Dynx+nz[j]*Dznx)-(Dynz-Dzny)*(nx[j]*Dxnz+ny[j]*Dynz+nz[j]*Dznz));
    hz[j] += K3mK2*(nx[j]*(nx[j]*Dxxnz+ny[j]*Dxynz+nz[j]*Dxznz)+ny[j]*(nx[j]*Dxynz+ny[j]*Dyynz+nz[j]*Dyznz)+nz[j]*(nx[j]*Dxznz+ny[j]*Dyznz+nz[j]*Dzznz)
        +(Dxnx+Dyny+Dznz)*(nx[j]*Dxnz+ny[j]*Dynz+nz[j]*Dznz)
        -(Dznx-Dxnz)*(nx[j]*Dxnx+ny[j]*Dynx+nz[j]*Dznx)-(Dzny-Dynz)*(nx[j]*Dxny+ny[j]*Dyny+nz[j]*Dzny));

    // remove part parallel to director
    hdotn = nx[j]*hx[j] + ny[j]*hy[j] + nz[j]*hz[j];
    hx[j] -= nx[j]*hdotn;
    hy[j] -= ny[j]*hdotn;
    hz[j] -= nz[j]*hdotn;

#if BC // Dirichlet boundary conditions along z
    if (m==0) {hx[j] = 0.0; hy[j] = 0.0; hz[j] = 0.0;}
    if (m==Lz-1) {hx[j] = 0.0; hy[j] = 0.0; hz[j] = 0.0;}
#endif
  }
  k=l=m=0;

# if SOFTBC
#pragma omp for
for (j=0; j<LL; j++) 
{
    // whats our 3D position?
    k = i_vr(j);
    l = j_vr(j);
    m = k_vr(j);

    // define neighbouring nodes
    xup=j+1; xdwn=j-1; yup=j+Lx; ydwn=j-Lx; zup=j+Lx*Ly; zdwn=j-Lx*Ly;
    xyuu=j+1+Lx; xyud=j+1-Lx; xydu=j-1+Lx; xydd=j-1-Lx;
    xzuu=j+1+Lx*Ly; xzud=j+1-Lx*Ly; xzdu=j-1+Lx*Ly; xzdd=j-1-Lx*Ly;
    yzuu=j+Lx+Lx*Ly; yzud=j+Lx-Lx*Ly; yzdu=j-Lx+Lx*Ly; yzdd=j-Lx-Lx*Ly;

    if(m==0)
    {
      Dznx = (nx[zup]-nx[j]);
      Dzny = (ny[zup]-ny[j]);
      Dznz = (nz[zup]-nz[j]);
      // calculate molecular field
      hx[j] = (K2*Dznx+K2*q0*ny[j]);
      hy[j] = (K2*Dzny-K2*q0*nx[j]);
      hz[j] = K2*Dznz+W*nz[j];
    }
    if(m==Lz-1)
    {
      Dznx = (nx[j]-nx[zdwn]);
      Dzny = (ny[j]-ny[zdwn]);
      Dznz = (nz[j]-nz[zdwn]);
      // calculate molecular field
      hx[j] =-(K2*Dznx+K2*q0*ny[j]);
      hy[j] =-(K2*Dzny-K2*q0*nx[j]);
      hz[j] =-K2*Dznz+W*nz[j];
    }

    // remove part parallel to director
    hdotn = nx[j]*hx[j] + ny[j]*hy[j] + nz[j]*hz[j];
    hx[j] -= nx[j]*hdotn;
    hy[j] -= ny[j]*hdotn;
    hz[j] -= nz[j]*hdotn;
}
# endif //SOFTBC

  // do the update -- first order Euler
#pragma omp for
  for (j=0; j<LL; j++)
  {
    nx[j] += Gamma*hx[j]*dt;
    ny[j] += Gamma*hy[j]*dt;
    nz[j] += Gamma*hz[j]*dt;
    //normalise
    sqrtndotn = sqrt(nx[j]*nx[j] + ny[j]*ny[j] + nz[j]*nz[j]);
    nx[j] /= sqrtndotn;
    ny[j] /= sqrtndotn;
    nz[j] /= sqrtndotn;
  }

}//end update

void writeStatistics(FILE* filestats)
{
  double totaltwist=0; 
  double totaltwistsq=0; 
  double totalsplaysqbendsq=0 ;

  double twistxzplane=0; 
  double twistsqxzplane=0; 
  double splaysqbendsqxzplane=0 ;

  for(int l=0;l<LL;l++)
  {
    double splaysq,twist,bendsq,twistsq;
    SplayTwistBendDensities(l, splaysq, twist,bendsq,twistsq);
    totaltwist+=twist; 
    totaltwistsq+=twistsq; 
    totalsplaysqbendsq+=bendsq+splaysq ;

    if((j_vr(l)==(Ny-1)/2))
    {
      twistxzplane += twist;
      twistsqxzplane += twistsq;
      splaysqbendsqxzplane += splaysq+bendsq;
    }   
  }

  double cholestericfreeenergy =K2*q0*totaltwist; 
  double nematicfreeenergy = 0.5*(K1mK2+K2)*totalsplaysqbendsq+0.5*K2*totaltwistsq;

  double cholestericfreeenergyxzplane =K2*q0*twistxzplane; 
  double nematicfreeenergyxzplane = 0.5*(K1mK2+K2)*splaysqbendsqxzplane+0.5*K2*twistsqxzplane;

  double radius,leftedge,rightedge,height;
  CalculateHopfionDimensionsXAxis(&radius,&leftedge,&rightedge,&height);

  fprintf(filestats, "%s %e %e %e %e %e %e %e %e %e %e %e %e\n",std::to_string(n).c_str(),nematicfreeenergy,cholestericfreeenergy,nematicfreeenergy+cholestericfreeenergy,nematicfreeenergyxzplane,cholestericfreeenergyxzplane,twistxzplane,splaysqbendsqxzplane,twistsqxzplane,radius,leftedge,rightedge,height); 
}  

void SplayTwistBendDensities(int j, double& splaysq, double &twist, double &bend, double &twistsq)
{
  int k,l,m;
  int xup,xdwn,yup,ydwn,zup,zdwn;
  double Dxnx,Dynx,Dznx,Dxxnx,Dyynx,Dzznx,Dxynx,Dxznx,Dyznx;
  double Dxny,Dyny,Dzny,Dxxny,Dyyny,Dzzny,Dxyny,Dxzny,Dyzny;
  double Dxnz,Dynz,Dznz,Dxxnz,Dyynz,Dzznz,Dxynz,Dxznz,Dyznz;

  int Lx=Nx;
  int Ly=Ny;
  int Lz=Nz;

  k=l=m=0;
  // whats our 3D position?
  k = i_vr(j);
  l = j_vr(j);
  m = k_vr(j);

  // define neighbouring nodes
  xup=j+1; xdwn=j-1; yup=j+Lx; ydwn=j-Lx; zup=j+Lx*Ly; zdwn=j-Lx*Ly;

  // correct for periodic boundaries
  if (k==0) {xdwn+=Lx; }
  if (k==Lx-1) {xup-=Lx; }
  if (l==0) {ydwn+=Lx*Ly; }
  if (l==Ly-1) {yup-=Lx*Ly;}
  if (m==0) {zdwn+=LL; }
  if (m==Lz-1) {zup-=LL;}

#if BC // cheap fix for the Dirichlet boundary conditions
  if (m==0) {xup=j; xdwn=j; yup=j; ydwn=j; zup=j; zdwn=j;}
  if (m==Lz-1) {xup=j; xdwn=j; yup=j; ydwn=j; zup=j; zdwn=j;}
#endif

  // calculate first order derivatives
  Dxnx = (nx[xup]-nx[xdwn])/2.0;
  Dynx = (nx[yup]-nx[ydwn])/2.0;
  Dznx = (nx[zup]-nx[zdwn])/2.0;

  Dxny = (ny[xup]-ny[xdwn])/2.0;
  Dyny = (ny[yup]-ny[ydwn])/2.0;
  Dzny = (ny[zup]-ny[zdwn])/2.0;

  Dxnz = (nz[xup]-nz[xdwn])/2.0;
  Dynz = (nz[yup]-nz[ydwn])/2.0;
  Dznz = (nz[zup]-nz[zdwn])/2.0;

  double bx =nx[j]*Dxnx + ny[j]*Dynx + nz[j]*Dznx; 
  double by =nx[j]*Dxny + ny[j]*Dyny + nz[j]*Dzny; 
  double bz =nx[j]*Dxnz + ny[j]*Dynz + nz[j]*Dznz; 
  bend =bx*bx+by*by+bz*bz;

  double splay = (Dxnx + Dyny +Dznz);
  splaysq =splay*splay;
  twist = nx[j]*(Dynz-Dzny)+ny[j]*(Dznx-Dxnz)+nz[j]*(Dxny-Dynx);
  twistsq = twist*twist;
  return;
}

void CalculateHopfionDimensionsXAxis(double* radius,  double* leftedge, double* rightedge, double* height)
{
  double tol = 0.05;
  float dir[4];
  int jmid = (Ny-1)/2;
  int kmid = (Nz-1)/2;

  int istart = 0;

  int isol=0;
  int ileftedge = 0;
  int irightedge = 0;
  int kheight = 0;

  for(int i=0;i<Nx;i++)
  {
    int l = Nx*Ny*kmid+Nx*jmid+i; 
    double tempnz = nz[l]; 
    if(tempnz < -1+tol) {ileftedge = i;break;}
  }
  // ileftedge contains the left edge of the z spike
  for(int i=ileftedge;i<Nx;i++)
  {
    int l = Nx*Ny*kmid+Nx*jmid+i; 
    double tempnz = nz[l]; 
    if((i!=ileftedge) && tempnz > -1+tol) {irightedge = i;break;}
  }
  // irightedge contains the left edge of the z spike
  isol =(ileftedge+irightedge)/2;

  // now that we have the i value of its height, lets scan left and right to get the edges, defined as where n drops to 0
  for(int i=isol;i<Nx;i++)
  {
    int l = Nx*Ny*kmid+Nx*jmid+i; 
    // grab the director 
    double tempnz = fabs(nz[l]);
    if(tempnz <tol) {irightedge = i;break;}
  }
  for(int i=isol;i>0;i--)
  {
    int l = Nx*Ny*kmid+Nx*jmid+i; 
    // grab the director 
    double tempnz = fabs(nz[l]);
    if(tempnz <tol) {ileftedge = i;break;}
  }

  // now lets scan up
  for(int k=kmid;k<Nz;k++)
  {
    int l = Nx*Ny*k+Nx*jmid+isol; 
    // grab the director 
    double tempnz = fabs(nz[l]);
    if(tempnz <tol) {kheight = k;break;}
  }

  *radius =fabs(isol-((Nx-1)/2)); 
  *leftedge =fabs(ileftedge-isol); 
  *rightedge =fabs(irightedge-isol); 
  *height =fabs(kheight-kmid); 
}


//Line number in the x direction. Range is from 0 to I-1.
int i_vr(int l){
  return l%Nx; 
}

//Line number in the y direction
int j_vr(int l){
  return (l%(Nx*Ny))/Nx; 
}

//Line number in the z direction
int k_vr(int l){
  return l/(Nx*Ny); 
}

inline void cross(double ax,double ay,double az,double bx,double by,double bz,double &rx,double &ry,double &rz)
{
  rx = ay*bz-az*by;
  ry = az*bx-ax*bz;
  rz = ax*by-ay*bx;
}
