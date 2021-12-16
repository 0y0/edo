// collection of commonly used objects (requires edo.scad)

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

// a basic case based on a profile, children will be placed on the bottom of bin
// h=height, d=lid height, t=thickness (outwards, can be negative), g=gap, sp=spacer
// view={undef:print, 0:cross-section, 1:bin, 2:lid, 3:closed, 4:profile}
module basic_case(profile, h=25, d=8, t=2.8, g=0.1, sp=0, top=true, bottom=true, view) {
  a = abs(t);
  h = max(a*4-0.2, h);
  d = max(a*2, min(h-a, d));
  o = encircle2d(profile);
  e = o[1]*2; // diameter of profile
  p = shift2d(profile, -o[0]); // center profile
  r = 0.4; // lower stopper radius
  u = 0.5; // upper stopper radius

  // lid cross section
  c1 = append(concat2d([
    step2d([[0,a-d],[0.001,0],[a/2-0.001,0],[0,d-a/2]]),
    ccw_path([0,0], [-a/2+0.01,a/2], $fs=$fs/2)
  ], [t/2,-a]), [t/2,0]);

  // bin cross section
  c2 = concat2d([
    step2d([[0.001,0],[g-0.001,0]]), // extra point for gap
    ccw_path([0,0], [a,a]),
    step2d([[0,h-d-a-0.2],[-a/2-g,0],[0,d-a-1.2]]),
    ccw_path([0,0], [-a/2,0], po=[-a/4,0], $fs=$fs/2),
    step2d([[0,-(h-2*a-1.4)]])
  ], [t<0?-a-g:-g,0]);

  if (view==undef) { // print
    scatter(e+abs(t)*4+sp) {
      basic_case(p, h, d, t, g, sp, top, bottom, 1) children(); // bin
      basic_case(p, h, d, t, g, sp, top, bottom, 2); // lid
    }
  } else child(view) {
    // cross-section
    clip([e*1.2,e*1.2,h], cy=0) {
      basic_case(p, h, d, t, g, sp, top, bottom, 1); // bin
      ascend(h-0.1) flipy() basic_case(p, h, d, t, g, sp, top, bottom, 2); // lid
    }
    // bin
    union() {
      m2 = [for (i=c2) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m2, loop=!bottom);
      ascend(t-0.01) children();
    }
    // lid
    flipy() {
      m1 = [for (i=c1) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m1, loop=!top);
    }
    // closed
    union() {
      basic_case(p, h, d, t, g, sp, top, bottom, 1); // bin
      ascend(h-0.1) flipy() basic_case(p, h, d, t, g, sp, top, bottom, 2); // lid
    }
    // profile only
    plot(p, color="red", dup=true);
  }
}

// a case based on a profile, with a cover on top for decorations, children will be placed on the bottom of bin
// h=height, d=lid height, t=thickness, s=cover thickness, b=border, g=gap, m=dimple diameter, sp=spacer
// view={undef:print 0:cross-section, 1:bin, 2:lid, 3:cover, 4:closed, 5:profile}
module cover_case(profile, h=25, d=8, t=2.8, s=1.2, b=2, g=0.1, m=0, sp=0, view) {
  a = abs(t);
  h = max(a*4-0.2, h);
  d = max(a*2, min(h-a, d));
  o = encircle2d(profile);
  e = o[1]*2; // diameter of profile
  c = o[0]; // centroid
  p = shift2d(profile, -c); // center profile
  r = 0.4; // lower stopper radius
  u = 0.5; // upper stopper radius

  // lid cross sections
  c1 = concat2d([
      step2d([[0,-a/2],[b,0],[0,-s+u-0.3]]),
      cw_path([0,0], [-u,-u]),
      step2d([[-r+u,0]]),
      cw_path([0,0], [r,-r]),
      step2d([[0,-d+a/2+s+r+0.3],[a/2,0],[0,d-a/2]]),
      ccw_path([0,0], [-a/2,a/2])
  ], [t/2-b,0]);

  // bin cross sections
  c2 = concat2d([
      step2d([[0.001,0],[g-0.001,0]]), // extra point for bottom
      ccw_path([0,0], [a,a]),
      step2d([[0,h-d-a-0.2],[-a/2-g,0],[0,d-a/2-s-1.2]]),
      ccw_path([0,0], [-a/2,0], po=[-a/4,0]),
      step2d([[0,-(h-2*a-1.4)]])
  ], [t<0?-a-g:-g,0]);
  
  if (view==undef) { // all parts
    scatter(e+abs(t)*4+sp) {
      cover_case(p, h, d, t, s, b, g, m, sp, 1) children(); // bin
      cover_case(p, h, d, t, s, b, g, m, sp, 2); // lid
      cover_case(p, h, d, t, s, b, g, m, sp, 3); // cover
    }
  } else child(view) {
    // cross-section
    clip([e*1.2,e*1.2,h], cy=0) {
      cover_case(p, h, d, t, s, b, g, m, sp, 1); // bin
      ascend(h-0.1) flipy() cover_case(p, h, d, t, s, b, g, m, sp, 2); // lid
      ascend(h-a/2-s-0.2) cover_case(p, h, d, t, s, b, g, m, sp, 3); // cover
    }
    // bin
    union() {
      m2 = [for (i=c2) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m2, loop=false);
      ascend(t-0.01) children();
    }
    // lid
    flipy() {
      m1 = [for (i=c1) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m1, loop=true);
    }
    // cover
    punch(locs=[c], m=m, zz=[s+a/2-0.2,10]) {
      solid(offset2d(profile, t/2-0.1), s);
      ascend(s-0.3) cookie_extrude(offset2d(profile, t/2-b-1), a/2+0.3, origin=c);
    }
    // closed
    union() {
      cover_case(p, h, d, t, s, b, g, m, sp, 1); // bin
      ascend(h-0.1) flipy() cover_case(p, h, d, t, s, b, g, m, sp, 2); // lid
      ascend(h-a/2-s-0.2) cover_case(p, h, d, t, s, b, g, m, sp, 3); // cover
    }
    // profile
    plot(p, color="red");
  }
}

