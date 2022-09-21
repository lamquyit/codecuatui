#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;


int tongchuso( long n ){
	int tong = 0;
	while (n>0){
		tong += n%10;
		n/=10;
	}
	int h = tong  ;
	if(h % 10 != h){
		return tongchuso(h);
	}else {
		return h;
	}
}

int main (){
	int m  ;
	cin >> m ;
	while (m--){
		long a ;
		cin >> a ;
		cout << tongchuso(a) << endl;
	}
}
