/*
    Copyright 2018
    Jack Binysh <j.binysh@warwick.ac.uk>,
    Gareth Alexander <g.p.alexadner@warwick.ack.uk>
    This software is provided under a BSD license. See LICENSE.txt.

    This file contains the global variables the user sets in the file "input.txt" or from command line
*/

#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
#include <vector>


// hard dirichlet boundary
//#define BC 0 // Dirichlet boundary conditions on (1) or off (0)
#define SOFTBC 1 // Dirichlet boundary conditions on (1) or off (0)

// do we want to read in a file? If so, whats its name?
enum InitialisationType {FROM_FUNCTION,FROM_FILE, FROM_SOLIDANGLE};
const InitialisationType InitialisationMethod = FROM_FUNCTION;

// simulation constants
const int Nmax = 200000;       // Number of timesteps
const int stepskip = 500;    // print director file every stepskip timesteps
const int stepskipstatistics = 100;    // print director file every stepskip timesteps
const double Gamma=1;
const double dt=0.1;
const double K2=1; // this is K2;

/* read in from config file */

// the input filename, in the form "xxxxx.txt"
extern std::string knot_filename;
// the output filepath
extern std::string output_dir;
// the number of knot components
extern int NumComponents;
// how many times should the curve be subdivided. If you dont want any, set this to 0
extern int NumRefinements;
// gridspacing
extern double h;
// array size in all 3 directions. For physical size, use Nx*h etc.
extern int Nx,Ny,Nz;
// Scaling sets whether the knot is scaled and centred
//ScaleProportionally ensures the aspect ratio of the knot is preserved during this process
extern bool Scaling, ScaleProportionally;
// what fraction of the physical box would you like the x,y,z dimensions of the knot to occupy? Note if ScaleProportionally is set, the minimum of these will be chosen.
extern double BoxFractionx, BoxFractiony, BoxFractionz;
// A user defined threshold between -1 and 0 (typically around -0.95). If one of the n . n_infty values across the knot is more negative than it, we will try another n_infty for calculation of the solid angle at some point
extern double Initialninftyx,Initialninftyy,Initialninftyz;
// The users initial choice of the vector n_infty. Normalise it!
extern std::vector<int> degrees;
extern double Threshold_ninf_dot_n;
// How thick the Hopfion Tube is
extern double TubeRadius;

#endif
