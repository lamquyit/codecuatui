#include <bits/stdc++.h>

using namespace std ;

int prime(long  n ){
	if(n < 2){
		return 0 ;
	}
	for (int i = 2 ; i <= sqrt(n);i++){
		if(n%i==0){
			return 0;
		}
	}
	return 1;
}

int main (){
	int m ;
	cin >> m ;
	while (m--){
		long a ;
		cin >> a ;
		int check = 0;
		for(int i = 2 ; i <= a/2; i++){
			if(prime(i) && prime(a-i)){
				cout << i << " " << a-i << endl;
				check = 1;
				break; 
			}
		}
	if(check == 0){
		cout << "-1\n"; 
		}
	}
}
