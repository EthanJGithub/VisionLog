import test from 'node:test';
import assert from 'node:assert/strict';
import { AdvancedTracker, assign } from '../src/webgpu/advancedTracker.js';
const detection = (x, confidence=.9, cls=0) => ({bbox_x:x,bbox_y:20,bbox_w:20,bbox_h:60,confidence,class_id:cls,class_label:cls?'car':'person'});
test('global assignment avoids the greedy collision',()=>assert.deepEqual(assign([[.1,.11],[.12,.9]]).sort(),[[0,1],[1,0]]));
test('two moving objects keep IDs when detector ranking reverses',()=>{
 const t=new AdvancedTracker();const initial=t.update([detection(10),detection(100)]);
 for(let n=1;n<12;n++) {const out=t.update([detection(100+n),detection(10+n)]);assert.equal(out.find(d=>d.bbox_x===10+n).track_id,initial[0].track_id);assert.equal(out.find(d=>d.bbox_x===100+n).track_id,initial[1].track_id);}
});
test('weak detections recover established tracks but cannot start new IDs',()=>{
 const t=new AdvancedTracker();const id=t.update([detection(10)])[0].track_id;t.update([detection(11)]);
 const result=t.update([detection(12,.15),detection(200,.15)]);assert.equal(result.length,1);assert.equal(result[0].track_id,id);
});
test('short occlusion retains identity without drawing invented boxes',()=>{
 const t=new AdvancedTracker();const id=t.update([detection(10)])[0].track_id;t.update([detection(12)]);assert.deepEqual(t.update([]),[]);assert.equal(t.update([detection(16)])[0].track_id,id);
});
test('expired tracks do not reuse IDs',()=>{
 const t=new AdvancedTracker({maxAge:2});const id=t.update([detection(10)])[0].track_id;for(let i=0;i<4;i++)t.update([]);assert.ok(t.update([detection(10)])[0].track_id>id);
});
test('class gating prevents assigning a person ID to a car',()=>{
 const t=new AdvancedTracker();const id=t.update([detection(10)])[0].track_id;assert.notEqual(t.update([detection(10,.9,1)])[0].track_id,id);
});
