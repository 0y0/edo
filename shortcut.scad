function GO(d, a) = [[0,0],d*[cos(a),sin(a)]]; // distance, angle
function CW(x, y) = cw_path([0,0],[x,y]);
function CCW(x, y) = ccw_path([0,0],[x,y]);
function CWC(x, y) = cw_path([0,0],[x,y],po=[x/2,y/2]);
function CCWC(x, y) = cw_path([0,0],[x,y],po=[x/2,y/2]);

// straight
function EE(d) = d<=0 ? [] : [[0,0],[d,0]];
function NN(d) = d<=0 ? [] : [[0,0],[0,d]];
function WW(d) = d<=0 ? [] : [[0,0],[-d,0]];
function SS(d) = d<=0 ? [] : [[0,0],[0,-d]];
// quarters
function NE(r) = r<=0 ? [] : CW(r,r);
function WN(r) = r<=0 ? [] : CW(-r,r);
function SW(r) = r<=0 ? [] : CW(-r,-r);
function ES(r) = r<=0 ? [] : CW(r,-r);
function EN(r) = r<=0 ? [] : CCW(r,r);
function NW(r) = r<=0 ? [] : CCW(-r,r);
function WS(r) = r<=0 ? [] : CCW(-r,-r);
function SE(r) = r<=0 ? [] : CCW(r,-r);
// halfs
function NES(d) = d<=0 ? [] : cw_path([0,0],[d,0],po=[d/2,0]);
function WNE(d) = d<=0 ? [] : cw_path([0,0],[0,d],po=[0,d/2]);
function SWN(d) = d<=0 ? [] : cw_path([0,0],[-d,0],po=[-d/2,0]);
function ESW(d) = d<=0 ? [] : cw_path([0,0],[0,-d],po=[0,-d/2]);
function ENW(d) = d<=0 ? [] : ccw_path([0,0],[0,d],po=[0,d/2]);
function NWS(d) = d<=0 ? [] : ccw_path([0,0],[-d,0],po=[-d/2,0]);
function WSE(d) = d<=0 ? [] : ccw_path([0,0],[0,-d],po=[0,-d/2]);
function SEN(d) = d<=0 ? [] : ccw_path([0,0],[d,0],po=[d/2,0]);
// parallels
function ENE(d, s=0) = d<=0 ? [] : concat2d([EN(d/2),NN(s),NE(d/2)]);
function ESE(d, s=0) = d<=0 ? [] : concat2d([ES(d/2),SS(s),SE(d/2)]);
function NWN(d, s=0) = d<=0 ? [] : concat2d([NW(d/2),WW(s),WN(d/2)]);
function NEN(d, s=0) = d<=0 ? [] : concat2d([NE(d/2),EE(s),EN(d/2)]);
function WNW(d, s=0) = d<=0 ? [] : concat2d([WN(d/2),NN(s),NW(d/2)]);
function WSW(d, s=0) = d<=0 ? [] : concat2d([WS(d/2),SS(s),SW(d/2)]);
function SWS(d, s=0) = d<=0 ? [] : concat2d([SW(d/2),WW(s),WS(d/2)]);
function SES(d, s=0) = d<=0 ? [] : concat2d([SE(d/2),EE(s),ES(d/2)]);

