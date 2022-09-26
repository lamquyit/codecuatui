#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

int prime(int n){
	for (int i = 2 ; i <= sqrt(n);i++){
		if(n%i == 0){
			return 0;
		}
	}
	return n > 1;
}

int main (){
	int m ;
	cin >> m ;
	while (m --){
		int a ;
		cin >> a ;
		for (int j = 2 ; j <= a ; j++){
			if(prime(j))
			cout << j << " ";
			}
		cout << endl;
		}
	}

