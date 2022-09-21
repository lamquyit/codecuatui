#include <stdio.h>
#include <math.h>
int songuyento(int n){
	for(int i=2;i<=sqrt(n);i++){
		if(n%i==0) return 0;
	}
	return n >1;
}
int main(){
	int n ;
	scanf("%d",&n);
	for(int i=1;i<n;i++){
		if(songuyento(i)) printf("%d ",i);
	}
	return 0;
}
