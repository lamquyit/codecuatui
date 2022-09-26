#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

int snt (long long n ){
	for(int i = 2; i <= sqrt(n); i++){
		if(n%i== 0){
			return 0 ;
		}
	}
	return n > 1;
}

int main(){
	int tc;
	long long n ;
	cin >> tc ;
	while (tc -- ){
		cin >> n ;
		for (int i = 2 ; i <= n ;i ++ ){
			if (snt (i)){
				cout << i << " ";
			}
		}
		cout << endl ;
	}
}
