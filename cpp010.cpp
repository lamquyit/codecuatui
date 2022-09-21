#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

int solocphat(long n ){
	while(n>0){
		int r = n % 10 ;
		if( ! ( r == 0 || r == 6|| r == 8)){
			return 0;
		}
		n/=10;
	}
	return 1;
}

int main(){
	int m ;
	cin >> m ;
	for (int i = 1 ; i <= m;i++){
		long a ;
		cin >> a ;
		if(solocphat(a)){
			cout<< "YES" << endl;
		}else {
			cout << "NO" << endl;
		}
	}
}
