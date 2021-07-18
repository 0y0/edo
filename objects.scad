// collection of commonly used objects (requires basic.scad)

module cell_generic(d, h, cd, ch, center, invert) {
    dz = center ? 0 : h/2;
    if (ch>0) beam([0,0,dz+(invert ? -h/2 : h/2-ch-0.01)], [0,0,dz+(invert ? -h/2+ch+0.01 : h/2)], cd); // anode cap
    beam([0,0,-h/2+dz+(invert ? ch : 0)], [0,0,h/2+dz-(invert ? 0 : ch)], d); // body
}

module cell_aa(d=14.5, h=50.5, cd=5.5, ch=1.5, center=false, invert=false) cell_generic(d, h, cd, ch, center, invert);
module cell_aaa(d=10.5, h=44.5, cd=3.5, ch=1.5, center=false, invert=false) cell_generic(d, h, cd, ch, center, invert);
module cell_cr123a(d=17, h=34.5, cd=5.5, ch=1, center=false, invert=false) cell_generic(d, h, cd, ch, center, invert);
module cell_18650(d=18, h=65, cd=10, ch=0.5, center=false, invert=false) cell_generic(d, h, cd, ch, center, invert);
module cell_32700(d=33, h=71, cd=15, ch=0.1, center=false, invert=false) cell_generic(d, h, cd, ch, center, invert);

module cell_spring_slot(s=2.1, t=0.5, b=1) {
    deepen(b+7) offset(t*3) line2d([[2,-t*2],[-2,-t*2]]);
    deepen(b+10) offset(t) line2d([[3.5,s],[6,s],[6,0],[-6,0],[-6,s],[-3.5,s]]);
    deepen(b) offset(t) line2d([[3.5,s],[3.5,0],[-3.5,0],[-3.5,s]]);
    ascend(b+5) sphere(t/2+1);
}

module bearing_608() {
    pipe(22, 7, m=8);
}

module bracket(dm, fold=0.5, r=5, a=90) {
    x = dm[0];
    y1 = dm[1] * fold;
    y2 = dm[1] - y1;
    z = dm[2];
    s1 = shift2d(spin2d(pad_path(x, y1, r, true), 180), [0,-y1/2]);
    solid(s1, z);
    s2 = shift2d(pad_path(x, y2, r, true), [0,y2/2]);
    ascend(z) flipx(a) ascend(-z) solid(s2, z);
    ascend(z) sweep(append(arc_path(z*2, [0,a]), [0,0]), [[x/2,0,0],[-x/2,0,0]]);
}

// a basic case based on a profile
// h=height, d=lid height, t=thickness (outwards, can be negative), g=gap
// view={undef:print, 0:cross-section, 1:bin, 2:lid, 3:profile}
module basic_case(profile, h=25, d=8, t=2.8, g=0.05, view) {
    a = abs(t);
    h = max(a*3, h);
    d = max(a*2, min(h-a, d));
    e = encircle2d(profile)[1]*2; // extent of profile
    c = avg(profile); // centroid of profile;
    r = 0.4; // lower stopper radius
    u = 0.5; // upper stopper radius

    // cross sections
    c1 = /* lid */ append(concat2d([
        step2d([[0.001,0],[a/2-0.001,0],[0,d-a/2]]),
        ccw_path([0,0], [-a/2+0.01,a/2], $fs=$fs/2)
    ], [t/2,-d]), [t/2,0]);
    c2 = /* bin */ concat2d([
        step2d([[0.001,0],[g-0.001,0]]), // extra point for bottom
        ccw_path([0,0], [a,a]),
        step2d([[0,h-d-a-0.2],[-a/2-g,0],[0,d-a-1.2]]),
        ccw_path([0,0], [-a/2,0], po=[-a/4,0], $fs=$fs/2)
    ], [t<0?-a-g:-g,0]);

    function q(d=0) = offset2d(profile, d);

    if (view==undef) { // print
        scatter(e+abs(t)*4) {
            basic_case(profile, h, d, t, g, 1); // bin
            basic_case(profile, h, d, t, g, 2); // lid
        }
    } else child(view) {
        // cross-section
        clip_xz(w=e*1.2, depth=e*1.2) {
            basic_case(profile, h, d, t, g, 1); // bin
            ascend(h-0.1) flipx() basic_case(profile, h, d, t, g, 2); // lid
        }
        // bin
        union() {
            sw = sweep_wall(c2, force3d(q(0)), loop=true);
            layered_block(sw, loop=true);
            solid(slice(sw, 1), a, scale=1.003);
        }
        // lid
        flipx() {
            sw = sweep_wall(c1, force3d(q(0)), loop=true);
            layered_block(sw, loop=true);
            solid(slice(sw, 1), -a, scale=1.003);
        }
        // profile only
        plot(profile, color="red");
    }
}

// a case based on a profile, with a cover on top for decorations
// h=height, d=lid height, t=thickness, s=cover thickness, b=border, g=gap, m=dimple diameter
// view={undef:print 0:cross-section, 1:bin, 2:lid, 3:cover, 4:profile}
module cover_case(profile, h=25, d=8, t=2.8, s=1.2, b=2, g=0.05, m=0, view) {
    a = abs(t);
    h = max(a*3, h);
    d = max(a*2, min(h-a, d));
    e = encircle2d(profile)[1]*2; // extent of profile
    c = avg(profile); // centroid of profile;
    r = 0.4; // lower stopper radius
    u = 0.5; // upper stopper radius

    // cross sections
    c1 = /* lid */ concat2d([
        step2d([[0,-a/2],[b,0],[0,-s+u-0.3]]),
        cw_path([0,0], [-u,-u], $fs=0.1),
        step2d([[-r+u,0]]),
        cw_path([0,0], [r,-r], $fs=0.1),
        step2d([[0,-d+a/2+s+r+0.3],[a/2,0],[0,d-a/2]]),
        ccw_path([0,0], [-a/2,a/2])
    ], [t/2-b,0]);
    c2 = /* bin */ concat2d([
        step2d([[0.001,0],[g-0.001,0]]), // extra point for bottom
        ccw_path([0,0], [a,a]),
        step2d([[0,h-d-a-0.2],[-a/2-g,0],[0,d-a/2-s-1.2]]),
        ccw_path([0,0], [-a/2,0], po=[-a/4,0])
    ], [t<0?-a-g:-g,0]);

    function q(d=0) = offset2d(profile, d);

    if (view==undef) { // all parts
        scatter(e+abs(t)*4) {
            cover_case(profile, h, d, t, s, b, g, m, 1); // bin
            cover_case(profile, h, d, t, s, b, g, m, 2); // lid
            cover_case(profile, h, d, t, s, b, g, m, 3); // cover
        }
    } else child(view) {
        // cross-section
        clip_xz(w=e*1.2, depth=e*1.2) {
            cover_case(profile, h, d, t, s, b, g, m, 1); // bin
            ascend(h-0.1) flipx() cover_case(profile, h, d, t, s, b, g, m, 2); // lid
            ascend(h-a/2-s-0.2) cover_case(profile, h, d, t, s, b, g, m, 3); // cover
        }
        // bin
        union() {
            sw = sweep_wall(c2, force3d(q(0)), loop=true);
            layered_block(sw, loop=true); // bin
            solid(slice(sw, 1), a, scale=1.003); // bottom
        }
        // lid
        flipx() layered_block(sweep_wall(c1, force3d(q(0)), loop=true), loop=true);
        // cover
        punch(locs=[c], m=m, zz=[s+a/2-0.2,10]) {
            solid(q(t/2-0.1), s);
            ascend(s-0.3) cookie_extrude(q(t/2-b-1), a/2+0.3, origin=c);
        }
        // profile
        plot(profile, color="red");
    }
}
