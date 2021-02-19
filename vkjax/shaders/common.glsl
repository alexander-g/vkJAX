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


uint shape_to_size(uint[N] shape){
    uint acc = 1;
    for(uint i=0; i<N; i++){
        acc *= shape[i];
    }
    return acc;
}

int[N] add_coords(int[N] a, int[N] b){
    int[N] c;
    for(uint i=0; i<N; i++){
        c[i] = a[i] + b[i];
    }
    return c;
}

int[N] sub_coords(int[N] a, uint[N] b){
    int[N] c;
    for(uint i=0; i<N; i++){
        c[i] = a[i] - int(b[i]);
    }
    return c;
}

int[N] multiply_coords(int[N] a, uint[N] b){
    int[N] c;
    for(uint i=0; i<N; i++){
        c[i] = a[i] * int(b[i]);
    }
    return c;
}