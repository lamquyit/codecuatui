#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std  ;

int main (){
	long long n;
	cin >> n;
	float tong ;
	if( n > 0 ){
		for(int i = 1; i <= n ;i++) {
			tong += 1.0 * 1/n;
		}
		cout << fixed << setprecision(4) << tong ;
	}
	return 0 ;
}
