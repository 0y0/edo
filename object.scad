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

// cell contact plate slot, s=slot gap, t=thickness, b=bottom gap, f=spine size
module cell_spring_slot(s=2.1, t=1, b=1, f=1) {
  d = t/2;
  c = s+t-1;
  slide(y=-t/2) {
    if (f>0) slide(y=(-f-t)/2) flipz() pad(6, f, h=b+7, r=1, half=true);
    deepen(b+10) offset(t/2) line2d([[3+d,c],[5.5+d,c],[5.5+d,0],[-5.5-d,0],[-5.5-d,c],[-3-d,c]]);
    deepen(b) offset(t/2) line2d([[3+d,s+t-1],[3+d,0],[-3-d,0],[-3-d,s+t-1]]);
    ascend(b+5) slide(y=(t-1)/2) flipx(-90) dome(2.5);
  }
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
// h=outside height, d=lid height, t=thickness (outwards, can be negative), g=lid gap, b=mouth spacing, sp=spread
// view={undef:print, 0:cross-section, 1:bin, 2:lid, 3:closed, 4:debug}
module basic_case(profile, h=25, d=10, t=2.8, g=0, b=0, sp=0, top=true, bottom=true, view) {
  a = abs(t);
  hh = max(a*4-0.2, h-a*2); // internal height
  dd = max(a*2, min(hh+a-0.2, d));
  bb = min(dd-a*2, b);
  o = encircle2d(profile);
  e = o[1]*2; // diameter of profile
  p = shift2d(profile, -o[0]); // center profile
  r = 0.4; // lower stopper radius
  u = 0.5; // upper stopper radius

  // lid cross section
  c1 = concat2d([
    step2d([if (b>0) [0,-bb],[(a+g)/2,0],[0,a-dd+bb+0.1],[(a-g)/2,0],[0,dd-a-0.1]]),
    ccw_path([0,0], [-a,a], $fs=$fs/2)
  ], [t<0?t:0,-a]);

  // bin cross section
  c2 = concat2d([
    ccw_path([0,0], [a,a]),
    step2d([[0,hh-dd+a-0.1],[-(a+g)/2,0],[0,dd-a-bb-1.1]]),
    ccw_path([0,0], [-(a-g)/2,0], po=[-(a-g)/4,0], $fs=$fs/2),
    step2d([[0,bb-(hh-1.4)]])
  ], [t<0?-a:0,0]);

  if (view==undef) { // print
    scatter(e+10+sp) {
      basic_case(p, h, d, t, g, b, sp, top, bottom, 1) if ($children>0) children(0); // bin
      basic_case(p, h, d, t, g, b, sp, top, bottom, 2) if ($children>1) children(1); // lid
    }
  } else select(view) {    // 1 bin + children
    // 0 cross-section
    clip([e*1.2,e*1.2,h], cy=0) {
      basic_case(p, h, d, t, g, b, sp, top, bottom, 1) if ($children>0) children(0); // bin
      ascend(h) flipy() basic_case(p, h, d, t, g, b, sp, top, bottom, 2) if ($children>1) children(1); // lid
    }
    // 1 bin + children
    union() {
      m2 = [for (i=c2) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m2, loop=!bottom);
      ascend(t-0.01) children();
    }
    // 2 lid - children
    flipy() {
      m1 = [for (i=c1) force3d(offset2d(profile, i[0]), i[1])];
      difference() {
        layered_block(m1, loop=!top);
        children();
      }
    }
    // 3 closed
    union() {
      basic_case(p, h, d, t, g, b, sp, top, bottom, 1) if ($children>0) children(0); // bin
      flipy(h=h-0.1) basic_case(p, h, d, t, g, b, sp, top, bottom, 2) if ($children>1) children(1); // lid
    }
    // 4 debug
    union() {
      color("red", 0.2) %plot(p, dup=true);
      plot(shift2d(c1, [0,h/2]), 0.1, color="gold");
      plot(shift2d(c2, [0,-h/2]), 0.1,  color="cyan");
    }
  }
}

// a case based on a profile, with a cover on top for decorations, children will be placed on the bottom of bin
// h=height, d=lid height, t=thickness, s=cover thickness, b=border, g=gap, m=dimple diameter, sp=spacer
// view={undef:print 0:cross-section, 1:bin, 2:lid, 3:cover, 4:closed, 5:debug}
module cover_case(profile, h=25, d=8, t=2.8, s=1.2, b=2, g=0, m=0, sp=0, rise=false, view) {
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
  q = a/4;
  c1 = concat2d([
      step2d([[-q,-q],[0,q-a/2],[b,0],[0,-s+u-0.3]]),
      cw_path([0,0], [-u,-u]),
      step2d([[-r+u,0]]),
      cw_path([0,0], [r,-r]),
      step2d([[0,-d+a/2+s+r+0.4],[a/2,0],[0,d-a/2-0.1]]),
      ccw_path([0,0], [-a/2,a/2])
  ], [t<0?t/2-b+q:-b+q,0]);

  // bin cross sections
  c2 = concat2d([
      step2d([[0.001,0],[g-0.001,0]]), // extra point for bottom
      ccw_path([0,0], [a,a]),
      step2d([[0,h-d-a-0.1],[-a/2-g,0],[0,d-a/2-s-1.1]]),
      ccw_path([0,0], [-a/2,0], po=[-a/4,0]),
      step2d([[0,-(h-2*a-1.4)]])
  ], [t<0?-a-g:-t/2-g,0]);
  
  if (view==undef) { // all parts
    scatter(e+10+sp) {
      cover_case(p, h, d, t, s, b, g, m, sp, rise, 1) children(); // bin
      cover_case(p, h, d, t, s, b, g, m, sp, rise, 2); // lid
      cover_case(p, h, d, t, s, b, g, m, sp, rise, 3); // cover
    }
  } else child(view) {
    // 0 cross-section
    clip([e*1.2,e*1.2,h], cy=0) {
      cover_case(p, h, d, t, s, b, g, m, sp, rise, 1); // bin
      ascend(h-0.1) flipy() cover_case(p, h, d, t, s, b, g, m, sp, rise, 2); // lid
      ascend(h-a/2-s-0.2) cover_case(p, h, d, t, s, b, g, m, sp, rise, 3); // cover
    }
    // 1 bin
    union() {
      m2 = [for (i=c2) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m2, loop=false);
      ascend(t-0.01) children();
    }
    // 2 lid
    flipy() {
      m1 = [for (i=c1) force3d(offset2d(profile, i[0]), i[1])];
      layered_block(m1, loop=true);
    }
    // 3 cover
    punch(locs=[c], m=m, zz=[s+a/2-0.2,10]) {
      solid(offset2d(profile, -0.1), s-0.2);
      if (rise) ascend(s-0.3) cookie_extrude(offset2d(profile, -b-1), a/2+0.3, origin=c);
    }
    // 4 closed
    union() {
      cover_case(p, h, d, t, s, b, g, m, sp, rise, 1); // bin
      ascend(h-0.1) flipy() cover_case(p, h, d, t, s, b, g, m, sp, rise, 2); // lid
      ascend(h-a/2-s-0.2) cover_case(p, h, d, t, s, b, g, m, sp, rise, 3); // cover
    }
    // 5 debug
    union() {
      color("red", 0.2) %plot(p, dup=true);
      plot(shift2d(c1, [0,h/2]), 0.1, color="gold");
      plot(shift2d(c2, [0,-h/2]), 0.1,  color="cyan");
    }
  }
}

// a rectangular case with hinge, dm=[width,length,height,lid_height], t=thickness, r=rounding, f=fillet, e=hinge_height
// g=gap, view={undef:print 0:cross-section, 1:bin, 2:lid, 3:closed}
module hinge_case(dm, t=3, r=0, f=0, e=0, g=0, view) {
  if (view==undef) {
    scatter(x=dm[0]+10) {
      union() {
        hinge_bin(dm, t, r, f, e, g);
        ascend(t-0.01) children();
      }
      flipy(h=dm[2]+t*2) hinge_lid(dm, t, r, f, e);
    }
  }
  else child(view) {
    // 0 cross-section
    clip([dm[0]+t*3,dm[1]+t*3,dm[2]+t*3], cy=0) {
      color("wheat") hinge_bin(dm, t, r, f, e, g);
      color("pink") hinge_lid(dm, t, r, f, e);
      ascend(t-0.01) children();
    }
    // 1 bin
    union() {
      hinge_bin(dm, t, r, f, e, g);
      ascend(t-0.01) children();
    }
    // 2 lid
    flipy(h=dm[2]+t*2) hinge_lid(dm, t, r, f, e);
    // 3 closed
    union() {
      color("wheat") hinge_bin(dm, t, r, f, e, g);
      color("pink") hinge_lid(dm, t, r, f, e);
      ascend(t-0.01) children();
    }
  }
}

// a rectangular bin, dm=[width,length,height,lid_height], t=thickness, r=rounding, f=fillet, e=hinge_height, g=gap
module hinge_bin(dm, t=3, r=0, f=0, e=0, g=0) {
  rr = min(dm[0]/2+t-0.05, dm[1]/3, t+max(0.1,r));
  lh = ifundef(dm[3], 2);
  c = offset2d(pad_path(dm[0]+t*2, dm[1]+t*2, rr), -t, tidy=0);
  p = [dm[0]/2,0,dm[2]-lh+t];
  s1 = dm[1]*0.4+0.15;
  s2 = t+max(0,e);
  difference() {
    union() {
      fillet_tray(c, dm[2]-lh+t, t, r=min(lh+t-4,f));
      ascend(dm[2]-lh+t-0.21) wall(c, 1.01, 1.5-0.2, flat=false, $fs=$fs/2); // rim
    }
    translate(p+[-t/2+t,0,-s2+t/2]) {
      ascend(s2/2) cube([t+3,s1+g*2,s2+2.2], center=true);
      rotate([90,-90,0]) spin(-5) {
        solid(arch_path(t+0.3, $fn=32), s1+g*2, bottom=-s1/2-g);
        slide(y=-t-0.29) cube([t+0.3,t+0.3,s1+g*2], center=true);
      }
      y_clone([0,-s1/2,-0.1]) {
        sphere(d=1.72, $fn=32);
        beam([0,0,0.2], [-3,0,1.2], 0.8, $fn=32);
      }
    }
    translate([0.3-p[0],0,p[2]-1.7]) sphere(d=1.6, $fn=32); // dimple
    translate([-p[0]-t-9.8,0,p[2]]) rotate([0,25,0]) cylinder(d=20, 5); // thumb notch
  }
}

// a rectangular lid, dm=[width,length,height,lid_height], t=thickness, r=rounding, f=fillet, e=hinge_height
module hinge_lid(dm, t=3, r=0, f=0, e=0) {
  rr = min(dm[0]/2+t-0.05, dm[1]/3, t+max(0.1,r));
  lh = ifundef(dm[3], 2);
  ff = min(lh+t-4,f);
  c = offset2d(pad_path(dm[0]+t*2, dm[1]+t*2, rr), -t, tidy=0);
  p = [dm[0]/2,0,dm[2]-lh+t];
  s1 = dm[1]*0.4+0.15;
  s2 = t+max(0,e);
  g = dm[1]/2+t-rr-0.2;
  q = s2-t/2;
  difference() {
    flipx(h=dm[2]+t*2) fillet_tray(c, lh+t-0.1, t, r=ff);
    beam(p+[-t/2+t-0.01,-g,-q], p+[-t/2+t-0.01,g,-q], 2*norm([t/2,q])-0.05);
    ascend(dm[2]-lh+t-0.21) wall(offset2d(c, 1.5, tidy=0), 1.2, -1.5-0.1, flat=true); // rim space
  }
  // hinge
  translate([p[0]-t+t+0.01,0,dm[2]+t+0.01]) fillet_bar(s1, [-2,-2]);
  translate(p+[-t/2+t,0,-s2+t/2]) {
    rotate([90,180,0]) solid(arch_path(t, $fn=32), s1, bottom=-s1/2);
    ascend(t/2-0.01) slab([t,s1,e+lh+t-max(t,ff)]);
    y_clone([0,s1/2,0]) sphere(d=1.7, $fn=32);
  }
  // bulge
  translate(p+[-dm[0]-t*1.3,0,0.1]) intersection() {
    translate([t+t/2,0,0]) scale([0.7,1,0.6]) dome(t*4);
    translate([t/4-t,0,0]) cube(t*3, center=true);
  }
  // strut
  translate([0.4-p[0],0,p[2]-1.6]) sphere(d=1.5, $fn=32);
  translate([0.2-p[0],0,p[2]-0.1]) rotate([90,180,90]) solid(arch_path(5), 1.3);
  translate([-dm[0]/2-0.1,-2.5,p[2]+1]) cube([1.3+0.3,5,lh-0.9]);
  translate([1.49-p[0],0,p[2]+lh+0.01]) fillet_bar(5, [3,-3]);
}

