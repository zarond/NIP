//#define BZ_HAVE_STD
#include <blitz/array.h>
//#define Real double ;
typedef float Real;
//typedef double Real;

namespace NIP {
	//typedef blitz::TinyVector<long long int, 10> VecTimes;
	//typedef float Real;
	typedef blitz::TinyVector<int, 3> size3;
	typedef blitz::TinyVector<int, 2> size2;
	typedef blitz::TinyVector<int, 1> size1;

	typedef blitz::TinyVector<Real, 3> Vec3;
	typedef blitz::TinyVector<Real, 2> Vec2;
	typedef blitz::TinyVector<Real, 1> Vec1;

	typedef blitz::Array<Real, 4> Array4D;
	typedef blitz::Array<Real, 3> Array3D;
	typedef blitz::Array<Real, 2> Array2D;
	typedef blitz::Array<Real, 1> Array1D;

}