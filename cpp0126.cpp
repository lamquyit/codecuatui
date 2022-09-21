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
	int m ;
	cin >> m ;
	while (m--){
	int  max , min ;
	cin >> min  >>  max ;
	for (int i = min ; i <= max ; i ++ ){
		if (SNT(i)){
				cout << i << " ";
			}
		}
		cout << endl;
	}
	return 0 ;
}
