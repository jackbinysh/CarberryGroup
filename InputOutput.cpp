#include "InputOutput.h"
#include "nematic.h"
#include "Constants.h"
#include "SolidAngle.h"

//Judge not...
int InitialiseSystemParameters(char*argv[])
{
  /* READ IN ALL GLOBAL DATA */
  double doverp=(double) atof(argv[1]);
  double K1overK2 = atof(argv[2]);
  double Wtemp = atof(argv[3]);
  Nx=(int) atoi(argv[4]);
  Ny=(int) atoi(argv[5]);
  Nz=(int) atoi(argv[6]);
  LL=Nx*Ny*Nz;
  output_dir = string(argv[7]);
  knot_filename= string(argv[8]);
  // set the elastic constants
  K1mK2 = (K1overK2)*K2 - K2;
  // K1 and K3 are the same
  K3mK2 =K1mK2;
  // set the chirality
  q0=(2.f*M_PI/(float)(Nz))*doverp;
  RunName = "doverp_"+std::to_string(doverp)+"_k3overk2_"+std::to_string(K1overK2)+"_Nx_"+std::to_string(Nx)+"_Ny_"+std::to_string(Ny)+"_Nz_"+std::to_string(Nz);
  W=Wtemp*(doverp)/((float)(Nz));
  return 0;
}

int InitialiseSolidAngle()
{
    /* READ IN SOLID ANGLE SETTINGS */

    ifstream CurveInputStream;
    CurveInputStream.open(("Knots/" + knot_filename + "/" + "input.txt").c_str());
    if(CurveInputStream.is_open())
    {

      string buff,buff2;
      stringstream ss;

      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> NumComponents;
      }
      degrees.resize(NumComponents);
      ss.str("");
      ss.clear();

      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> NumRefinements;
      }
      ss.str("");
      ss.clear();

      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> Scaling;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> ScaleProportionally;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> BoxFractionx;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> BoxFractiony;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> BoxFractionz;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> Initialninftyx;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> Initialninftyy;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> Initialninftyz;
      }
      ss.str("");
      ss.clear();
      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> Threshold_ninf_dot_n;
      }
      ss.str("");
      ss.clear();

      for(int i =0;i<NumComponents;i++)
      {
        if(getline(CurveInputStream,buff))
        {
          ss << buff;
          getline(ss,buff2,'=');
          ss >> degrees[i];
        }
        ss.str("");
        ss.clear();
      }

      if(getline(CurveInputStream,buff))
      {
        ss << buff;
        getline(ss,buff2,'=');
        ss >> TubeRadius;
      }
      ss.str("");
      ss.clear();
    }
    else
    {
      std::cerr << "Could knot find a system parameters input file for the given knot name. Aborting";
      return 1;
    }
    return 0;
}


int InitialiseFromFile(Link& Curve)
{
  Curve.NumPoints = 0;
  Curve.NumComponents = NumComponents;
  Curve.Components.resize(NumComponents);

  double maxxin = 0;
  double maxyin = 0;
  double maxzin = 0;
  double minxin = 0;
  double minyin = 0;
  double minzin = 0;

  for(int i=0;i<Curve.NumComponents;i++)
  {
    stringstream ss;
    string buff,filename;

    ss.clear();
    ss.str("");
    if (Curve.NumComponents==1)
    {
      ss << "Knots/" << knot_filename << "/" << knot_filename << ".txt";
    }
    else
    {
      ss << "Knots/" << knot_filename << "/" <<  knot_filename << "_" << i << ".txt";
    }

    filename = ss.str();
    ifstream CurveInputStream;
    CurveInputStream.open(filename.c_str());
    if(CurveInputStream.is_open())
    {
      while(CurveInputStream.good())
      {
        double xcoord,ycoord,zcoord;
        if(getline(CurveInputStream,buff))
        {
          ss.clear();
          ss.str("");
          ss << buff;
          ss >> xcoord >> ycoord >> zcoord;
        }
        else break;

        knotpoint Point;
        Point.xcoord = xcoord;
        Point.ycoord = ycoord;
        Point.zcoord = zcoord;
        Curve.Components[i].knotcurve.push_back(Point);

        if(xcoord>maxxin) maxxin = xcoord;
        if(ycoord>maxyin) maxyin = ycoord;
        if(zcoord>maxzin) maxzin = zcoord;
        if(xcoord<minxin) minxin = xcoord;
        if(ycoord<minyin) minyin = ycoord;
        if(zcoord<minzin) minzin = zcoord;
      }

      CurveInputStream.close();

      Curve.NumPoints += Curve.Components[i].knotcurve.size();
    }
    else
    {
      std::cerr << "Could not find all input knot filenames in the folder with the given knot name. Aborting";
      return 1;
    }
  }

  Curve.minx=minxin;
  Curve.maxx=maxxin;
  Curve.miny=minyin;
  Curve.maxy=maxyin;
  Curve.minz=minzin;
  Curve.maxz=maxzin;
  return 0;
}

int InitialiseFromFileNP(Link& CurveNP)
{
  CurveNP.NumPoints = 0;
  CurveNP.NumComponents = 1;//NumComponents;
  CurveNP.Components.resize(1);//(NumComponents);

  for(int i=0;i<CurveNP.NumComponents;i++)
  {
    stringstream ss;
    string buff,filename;

    ss.clear();
    ss.str("");
    if (CurveNP.NumComponents==1)
    {
      ss << "Knots/" << knot_filename << "/" << knot_filename << "NP.txt";
    }
    else
    {
      ss << "Knots/" << knot_filename << "/" <<  knot_filename << "NP_" << i << ".txt";
    }

    filename = ss.str();
    ifstream CurveInputStream;
    CurveInputStream.open(filename.c_str());
    if(CurveInputStream.is_open())
    {
      while(CurveInputStream.good())
      {
        double xcoord,ycoord,zcoord;
        if(getline(CurveInputStream,buff))
        {
          ss.clear();
          ss.str("");
          ss << buff;
          ss >> xcoord >> ycoord >> zcoord;
        }
        else break;

        knotpoint Point;
        Point.xcoord = xcoord;
        Point.ycoord = ycoord;
        Point.zcoord = zcoord;
        CurveNP.Components[i].knotcurve.push_back(Point);
      }

      CurveInputStream.close();

      CurveNP.NumPoints += CurveNP.Components[i].knotcurve.size();
    }
    else
    {
      std::cerr << "Could not find all input knot filenames in the folder with the given knot name. Aborting";
      return 1;
    }
  }

  return 0;
}

void OutputSolidAngle(const Link& Curve,const vector<double>& omega,const string filename)
{
  string fn = "Knots/" + knot_filename + "/" + filename+".vtk";
  ofstream Aout (fn.c_str());
  Aout << "# vtk DataFile Version 3.0\nKnot\nASCII\nDATASET STRUCTURED_POINTS\n";
  Aout << "DIMENSIONS " << Nx << ' ' << Ny << ' ' << Nz << '\n';
  Aout << "ORIGIN " << x(0) << ' ' << y(0) << ' ' << z(0) << '\n';
  Aout << "SPACING " << 1 << ' ' << 1 << ' ' << 1 << '\n';
  Aout << "POINT_DATA " << Nx*Ny*Nz << '\n';
  Aout << "SCALARS omega float\nLOOKUP_TABLE default\n";
  for(int k=0; k<Nz; k++)
  {
    for(int j=0; j<Ny; j++)
    {
      for(int i=0; i<Nx; i++)
      {
        int n = pt(i,j,k);
        Aout << omega[n] << '\n';
      }
    }
  }
  Aout.close();
}

void OutputScaledKnot(Link& Curve)
{
  for( int c=0; c < NumComponents ; c++)
  {
    stringstream ss;
    ss.str("");
    ss.clear();

    ss << "Knots/" << knot_filename << "/" << "ParaViewBoundaryCurve_" << knot_filename << "_" << c<< ".obj";
    ofstream knotout (ss.str().c_str());

    int i;
    int n = Curve.Components[c].knotcurve.size();

    for(i=0; i<n; i++)
    {
      knotout  <<"v " <<  Curve.Components[c].knotcurve[i].xcoord << ' ' << Curve.Components[c].knotcurve[i].ycoord << ' ' << Curve.Components[c].knotcurve[i].zcoord << '\n';
    }
    for(i=1; i<n; i++)
    {
      knotout << "l " << i << ' ' << incp(i,1,n+1) << '\n';
    }
    knotout << "l " << n  << ' ' << 1  << '\n';
    knotout.close();
  }

}

void OutputScaledKnotNP(Link& CurveNP)
{
  stringstream ss;
  ss.str("");
  ss.clear();

  ss << "Knots/" << knot_filename << "/" << "ParaViewBoundaryCurve_" << knot_filename << "NP.obj";
  ofstream knotout (ss.str().c_str());

  int i;
  int n = CurveNP.Components[0].knotcurve.size();

  for(i=0; i<n; i++)
  {
    knotout  <<"v " <<  CurveNP.Components[0].knotcurve[i].xcoord << ' ' << CurveNP.Components[0].knotcurve[i].ycoord << ' ' << CurveNP.Components[0].knotcurve[i].zcoord << '\n';
  }
  for(i=1; i<n; i++)
  {
    knotout << "l " << i << ' ' << incp(i,1,n+1) << '\n';
  }
  knotout << "l " << n  << ' ' << 1  << '\n';
  knotout.close();
}

void writeVTKFiles()
{
  // now lets write things, in binary 
  FILE *file;
  string fn = output_dir + "/"+RunName+"_"+to_string(n)+".vtk";
  file=fopen(fn.c_str(),"w");
  fprintf(file,"# vtk DataFile Version 3.0\nKnot\nBINARY\nDATASET STRUCTURED_POINTS\n");
  fprintf(file,"DIMENSIONS %d %d %d\n",Nx,Ny,Nz);
  fprintf(file,"ORIGIN 0 0 0\n");
  fprintf(file,"SPACING 1 1 1\n");
  fprintf(file,"POINT_DATA %d\n",LL);
  fprintf(file,"VECTORS n float\n");

  float temp[3];
  for(int l=0; l<LL; l++){
    temp[0]=nx[l];
    temp[1]=ny[l];
    temp[2]=nz[l];
    // go to bigendian
    float temp0 = FloatSwap(temp[0]); 
    temp[0]=temp0;
    float temp1 = FloatSwap(temp[1]);
    temp[1]=temp1;
    float temp2 = FloatSwap(temp[2]);
    temp[2]=temp2;

    fwrite(temp,sizeof(float),3,file);
  };
  fclose(file);


} // end writeVTKFiles

// Function to take a little-endian encoded float and make it big-endian (or vice versa). Useful for reading to paraview
float FloatSwap( float f )
{
  union
  {
    float f;
    char b[4];
  } dat1, dat2;

  dat1.f = f;
  dat2.b[0] = dat1.b[3];
  dat2.b[1] = dat1.b[2];
  dat2.b[2] = dat1.b[1];
  dat2.b[3] = dat1.b[0];
  return dat2.f;
}
