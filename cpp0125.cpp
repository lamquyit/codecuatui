#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

bool SNT(int n ){
	if (n<2){
		return 0 ;
	}
	for (int i = 2 ; i <= sqrt(n); i++){
		if (n%i == 0 ){
			return false;
		}
	}
	return true;
}

int main (){
	int a, b , max , min ;
	cin >> a >> b ;
	if (a>b){
		max = a ;
		min = b;
	}else {
		max = b ;
		min = a ;
	}
	for (int i = min ; i <= max ; i ++ ){
		if (SNT(i)){
			cout << i << " ";
		}
	}
	return 0 ;
}
