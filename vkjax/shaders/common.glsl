//this file gets included in other .comp shader files
//parent file must define: N (number of dimensions)

int[N] unravel_index(uint index, uint[N] shape){
    int[N] coords;
    for(int i=int(N)-1; i>=0; i--){
        coords[i] = int(index % shape[i]);
        index     = index / shape[i]; 
    }
    return coords;
}

uint ravel_coords(int[N] coords, uint[N] shape){
    uint index  = 0;
    uint stride = 1;
    for(int i=int(N)-1; i>=0; i--){
        index  += coords[i]*stride;
        stride *= shape[i];
    }
    return index;
}

bool is_out_of_bounds(int[N] coords, uint[N] shape){
    bool is_oob = false;
    for(uint i=0; i<N; i++){
        is_oob = is_oob || (coords[i]<0 || coords[i]>=shape[i]);
    }
    return is_oob;
}