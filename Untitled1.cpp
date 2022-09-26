#include <bits/stdc++.h>

using namespace std ;

long long ucln(long long a , long long b){
	if (b==0) return a ;
	else return ucln(b,a%b);
}

long long bcnn(long long a , long long b ){
	return a*b/ucln(b,a%b);
}

long long kq(int n ){
	long long a = 1 ;
	for ( long long i = 2 ; i <= n ; i ++ ){
		a = bcnn(a,i);
	}
	return a ;
}main (){
	int t ;
	cin >> t ;
	while (t--){
		int n ;
		cin >> n ;
		cout << kq(n)<< endl ;
	}
}


