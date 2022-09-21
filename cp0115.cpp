#include <iostream>
#include <math.h>

using namespace std;

int t;
long n ;

int main(){
	int t ;
	cin >> t;
	while(t--){
		cin>>n;
		for(int i = 2 ; i<=sqrt(n);i++){
			if(n%i == 0 ){
				int countSoMu=0;
				while (n%i==0){
					n/=i;
					++countSoMu;
				}
				cout << i << " " << countSoMu << " ";
			}
		}
		if(n>1){
			cout << n << " " << 1; 
		}
		cout<< endl;
	}
	return 0 ;
}
