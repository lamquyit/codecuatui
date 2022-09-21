#include <stdio.h>
#include <math.h>
int songuyento(long long n){
	for(int i=2;i<=sqrt(n);i++){
		if(n%i==0) return 0;
	}
	return n>1;
}
int main(){
	int a;
	scanf("%d",&a);
	for(int i=1;i<=a;i++){
		long long m;
		scanf("%lld",&m);
		if(songuyento(m)) printf("YES\n");
		else printf("NO\n");
		
	}
	
	return 0;
}
