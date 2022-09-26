#include <iostream>
#include <iomanip>
#include <math.h>
#include <strinG>

using namespace std;

int main (){
	char a [15]={'A','B','B','A','D','C','C','A','B','D','C','C','A','B','D'};
	char b [15]={'A','C','C','A','B','C','D','D','B','B','C','D','D','B','B'};
	int t ;
	cin >> t;
	while (t--){
		double tmp = 10;
		char check[15];
		int n ;
		cin >> n ;
		for( int i = 0 ; i < 15 ; i++){
			cin >> check [i];
		}
		for( int i = 0 ; i < 15 ; i++){
			if ( n == 101 ){
				if(check[i] != a[i]){
					tmp = tmp - (double) 10 /15;
				}
			}else if (n == 102 ){
				if(check[i] == b[i]){
					tmp = tmp - (double) 10 /15;
				}
			}
		}
		cout << fixed << setprecision(2) << tmp << endl ;
	}	
}
