// foo() is a user-defined object with d=diameter, t=inflation, top=true if top layer (useful for engraving)
// inflation should be uniform with respect to surface normals, not just scaling from a center point
module foo(d, t, top=false) {}

module foo_lid(d, cut, t, h, rim, flip=0, k=10) {
    flipx(flip==0 ? 0 : 180, flip) {  
        ascend(rim+0.1) clip_floor(0, d+t*2, depth=h+10) ascend(-cut) difference() {
            foo(d, 0, top=true);
            foo(d, -t);
        }
        intersection() {
            clip_floor(rim, d+t*2, depth=h+10) ascend(rim-cut) foo(d, 0);
            foo_slice(d, cut, [-t/2-0.2,-t-0.2], k);
        }
        *intersection() {
            clip_floor(rim, d+t*2) ascend(rim-cut) foo(d, 0);
            ascend(rim+0.1) foo_slice(d, cut, [-t,0], t-0.1);
        }
        foo_slice(d, cut, [-t/2-0.2,-t-0.2], rim);
    }
}

module foo_bin(d, cut, t, h, rim) {
    foo_slice(d, cut, [0,-t], h-rim-0.1);
    ascend(h-rim-0.1) foo_slice(d, cut, [-t/2,0], rim);
    deepen(t) projection(true) ascend(-cut) foo(d, 0);
}

module foo_slice(d, cut, tt, h) {
    deepen(h) difference() {
        offset(max(tt)) projection(true) ascend(-cut) foo(d, 0);
        offset(min(tt)) projection(true) ascend(-cut) foo(d, 0);
    }
}

module foo_case(d, cut, t, h, rim=3, flip=0, debug=false) {
    clip_xz(w=d*3, depth=d, enable=debug) scatter(debug ? 0 : d+10) {
        ascend(debug ? h-rim : 0) foo_lid(d, cut, t, h, rim, flip);
        foo_bin(d, cut, t, h, rim);
    }
}
