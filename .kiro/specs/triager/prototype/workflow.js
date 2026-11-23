/* Offline interaction-design fixture. All scores and judgements below are invented. */
"use strict";

const TriagerDemo = (() => {
  const IN = "in", OUT = "out", UNKNOWN = "not_evaluated";
  const date = "2026-11-18T";
  // The same original can occur on several hosts and in several time buckets.
  const fixture = [
    ["login denied for guest", .98, [["edge-01","08:01"],["edge-02","09:08"]]],
    ["login challenge failed", .96, [["edge-01","08:02"]]],
    ["authentication challenge rejected", .94, [["edge-01","08:05"],["edge-01","08:46"],["idp-01","09:17"]]],
    ["session verification declined", .91, [["idp-01","09:12"],["edge-02","10:15"]]],
    ["identity check expired", .88, [["idp-01","10:18"]]],
    ["token refresh scheduled", .84, [["edge-02","08:27"],["edge-02","09:42"]]],
    ["service account rotation complete", .82, [["idp-01","10:23"]]],
    ["identity response format changed", .80, [["edge-01","09:26"],["idp-01","11:04"]]],
    ["login succeeded for guest", .40, [["edge-02","08:50"],["edge-01","10:46"]]],
    ["login window closed", .31, [["edge-02","09:15"]]],
    ["session cache cleared", .22, [["idp-01","10:52"],["idp-01","11:19"]]],
    ["service heartbeat received", .10, [["edge-01","11:48"]]]
  ];
  const opinions = {
    R03: [IN, "Similar wording in a reviewed challenge rejection."],
    R04: [IN, "Reviewed session-verification failures provide a lead."],
    R05: [IN, "A reviewed expiry warrants a closer look."],
    R06: [OUT, "The fixture records a scheduled refresh."],
    R07: [OUT, "The fixture records a completed rotation."],
    R08: [UNKNOWN, "The fixture producer abstained on an unfamiliar format."]
  };
  const rows = fixture.map(([text,score],i) => ({id:"R"+String(i+1).padStart(2,"0"),text,score}));
  const events = fixture.flatMap((item,i) => item[2].map(([host,minute],j) => ({
    id:rows[i].id, host, time:date+minute+":00Z", locator:host+"/auth.log#"+(10*(i+1)+j+1)
  })));
  const all = events.map(() => IN);
  const bin = time => time.slice(0,13)+":00Z";
  const state = yes => yes ? IN : OUT;
  const union = xs => xs.includes(IN) ? IN : xs.includes(UNKNOWN) ? UNKNOWN : OUT;
  const intersect = xs => xs.includes(OUT) ? OUT : xs.includes(UNKNOWN) ? UNKNOWN : IN;
  const negate = x => x===IN ? OUT : x===OUT ? IN : UNKNOWN;
  const visible = mask => events.filter((_,i) => mask[i]===IN);
  const unique = xs => [...new Set(xs)];
  const ids = occurrences => unique(occurrences.map(e=>e.id));
  const copy = x => JSON.parse(JSON.stringify(x));
  function requireValue(condition, message) {
    if (!condition) throw new Error(message);
  }
  function distribution(states, unit) {
    const included = states.filter(x=>x===IN).length;
    const unknown = states.filter(x=>x===UNKNOWN).length;
    return {unit,parent:states.length,in:included,out:states.length-included-unknown,unknown,
      retained_share:states.length ? included/states.length : null};
  }
  function count(mask, parentMask) {
    const grouped = new Map(), states = [];
    events.forEach((e,i) => {
      if (parentMask[i]!==IN) return;
      const group = grouped.get(e.id) || [];
      group.push(mask[i]); grouped.set(e.id,group); states.push(mask[i]);
    });
    return {counts:distribution([...grouped.values()].map(union),"distinct_originals"),
      occurrence_counts:distribution(states,"occurrences")};
  }
  function matchesGlob(text, pattern) {
    // Dynamic programming keeps wildcard matching bounded, even for repeated stars.
    let previous = [true,...Array(text.length).fill(false)];
    for (const token of pattern) {
      const next = [token==="*" && previous[0]];
      for (let j=1; j<=text.length; j++) {
        next[j] = token==="*" ? previous[j] || next[j-1] : previous[j-1] && (token==="?" || token===text[j-1]);
      }
      previous = next;
    }
    return previous[text.length];
  }
  function seededKey(id, seed) {
    let value = 2166136261;
    for (const ch of seed+"/"+id) value = Math.imul(value ^ ch.charCodeAt(0),16777619);
    return value >>> 0;
  }
  function frequencies(occurrences, key) {
    const buckets = new Map();
    for (const e of occurrences) {
      const value=key(e), bucket=buckets.get(value) || [];
      bucket.push(e); buckets.set(value,bucket);
    }
    return [...buckets].sort(([a],[b])=>a.localeCompare(b)).map(([value,xs])=>({value,count:xs.length,originals:ids(xs).length}));
  }
  const regions = [
    ["a_minus_b","A without B",{difference:["a","b"]},"ab"],
    ["both","A and B",{intersection:["a","b"]},"ab"],
    ["b_minus_a","B without A",{difference:["b","a"]},"ab"],
    ["neither","Neither A nor B",{intersection:[{state:{set:"a",is:OUT}},{state:{set:"b",is:OUT}}]},"ab"],
    ["recommended","C recommended","c","c"],
    ["pass","C passed",{state:{set:"c",is:OUT}},"c"],
    ["unknown","C unevaluated",{state:{set:"c",is:UNKNOWN}},"c"]
  ];
  const seedArgs = [
    ["triager_open",{source:{ref:"auth-fixture"}}],
    ["triager_select",{scope:"U",producer:{kind:"score",model:"fixture-score",top_k:8,ties:"record_id_ascending"}}],
    ["triager_select",{scope:"U",producer:{kind:"filter",syntax:"glob",query:"*login*"}}],
    ["triager_compose",{mode:"view",snapshot:"U",scope:"U",sets:{a:"A",b:"B"},expr:{difference:["a","b"]}}],
    ["triager_select",{scope:"D",producer:{kind:"recommend",method:"issued_batch",action:"collect",
      run:"fixture-review",submissions_ref:"fixture-batch",membership:"recommend"}}]
  ];
  const allocation = {mode:"allocate",snapshot:"U",scope:"D",sets:{d:"D",c:"C"},
    expr:{intersection:["d","c"]},budget:{total:4,duplicate_rule:"first_pool_wins",shortfall:"leave_unfilled",pools:[
      {name:"general",limit:2,from:{intersection:["d","c"]},order:["score_desc","record_id_ascending"]},
      {name:"unfamiliar",limit:1,from:{intersection:["d",{state:{set:"c",is:UNKNOWN}}]},order:["record_id_ascending"]},
      {name:"audit",limit:1,from:{difference:["d","c"]},order:["seeded_hash","record_id_ascending"],seed:5}
    ]}};

  function create(store = {views:new Map(), cache:new Map(), receipts:new Map(), next:0}) {
    const trace = [];
    const get = handle => {
      requireValue(store.views.has(handle),"Unknown view "+handle);
      return store.views.get(handle);
    };
    function publish(request, parent, mask, title, preferred, more={}) {
      const key=JSON.stringify(request);
      if (store.cache.has(key)) return get(store.cache.get(key));
      const handle=preferred || "V"+(++store.next);
      requireValue(!store.views.has(handle),"Handle already bound: "+handle);
      const view={handle,title,parent,mask,...count(mask,parent ? get(parent).mask : all),...more};
      store.cache.set(key,handle); store.views.set(handle,view);
      return view;
    }
    function kind(v) {
      if (!v.parent) return "snapshot";
      if (v.allocations) return "allocation";
      if (v.producer) return "selection";
      return "view";
    }
    const summary = v => ({handle:v.handle,kind:kind(v),title:v.title,parent:v.parent,counts:v.counts,
      occurrence_counts:v.occurrence_counts,allocated:!!v.allocations,complete:true});
    const scoped = (parent, evaluate) => events.map((e,i)=>parent.mask[i]===IN ? evaluate(e,i) : UNKNOWN);
    function evaluate(expr, bindings, i, unpreparedIsUnknown=false) {
      if (typeof expr==="string") {
        if (unpreparedIsUnknown && !store.views.has(bindings[expr])) return UNKNOWN;
        return get(bindings[expr]).mask[i];
      }
      const operand=child=>evaluate(child,bindings,i,unpreparedIsUnknown);
      if (expr.state) return state(operand(expr.state.set)===expr.state.is);
      if (expr.intersection) return intersect(expr.intersection.map(operand));
      if (expr.union) return union(expr.union.map(operand));
      if (expr.difference) return intersect([operand(expr.difference[0]),negate(operand(expr.difference[1]))]);
      throw new Error("Unknown expression");
    }
    function members(e) {
      const i=events.indexOf(e), result={};
      for (const label of ["A","B","C"]) result[label.toLowerCase()]=store.views.get(label)?.mask[i] ?? UNKNOWN;
      return result;
    }
    function record(v, id) {
      const original=rows.find(x=>x.id===id);
      const inside=visible(v.mask).filter(e=>e.id===id);
      requireValue(inside.length && original,"Record is outside this view: "+id);
      return {...original,host:inside[0].host,time:inside[0].time,occurrences:inside.length,
        membership:members(inside[0]),allocation:v.allocations?.[id] ?? null};
    }
    function resultRecords(v) {
      const order=v.order || ids(visible(v.mask)).sort();
      return order.map(id=>record(v,id));
    }
    function select(args) {
      const parent=get(args.scope), p=args.producer;
      let mask,title,preferred;
      if (p.kind==="score") {
        requireValue(p.model==="fixture-score" && p.ties==="record_id_ascending" && Number.isInteger(p.top_k) && p.top_k>=0,"Invalid fixture scoring rule");
        const available=new Set(ids(visible(parent.mask)));
        const top=rows.filter(r=>available.has(r.id)).sort((a,b)=>b.score-a.score || a.id.localeCompare(b.id)).slice(0,p.top_k);
        const selected=new Set(top.map(r=>r.id));
        mask=scoped(parent,e=>state(selected.has(e.id)));
        title="Score · top "+p.top_k;
        if (args.scope==="U" && p.top_k===8) preferred="A";
      } else if (p.kind==="filter") {
        requireValue(["literal","glob"].includes(p.syntax) && typeof p.query==="string" && p.query.length<=120,"Use literal or glob, up to 120 characters");
        const selected=new Set(rows.filter(r=>p.syntax==="glob" ? matchesGlob(r.text,p.query) : r.text.includes(p.query)).map(r=>r.id));
        mask=scoped(parent,e=>state(selected.has(e.id))); title=p.query;
        if (args.scope==="U" && p.syntax==="glob" && p.query==="*login*") preferred="B";
      } else {
        requireValue(p.kind==="recommend" && args.scope==="D" && p.method==="issued_batch" &&
          p.action==="collect" && p.run==="fixture-review" && p.submissions_ref==="fixture-batch" &&
          p.membership==="recommend","Only the frozen fixture recommendations are available");
        mask=scoped(parent,e=>opinions[e.id][0]); title="C recommended"; preferred="C";
      }
      const v=publish(["select",args.scope,p.kind,p.model,p.top_k,p.syntax,p.query,p.run],args.scope,mask,title,preferred,{producer:p.kind});
      return {...summary(v),set:v.handle,coverage:{in:v.counts.in,out:v.counts.out,not_evaluated:v.counts.unknown}};
    }
    function compose(args) {
      requireValue(args.snapshot==="U","Incompatible snapshot");
      const parent=get(args.scope), mask=scoped(parent,(_,i)=>evaluate(args.expr,args.sets,i));
      if (args.mode==="allocate") {
        requireValue(JSON.stringify(args.budget)===JSON.stringify(allocation.budget) && args.scope==="D" &&
          args.sets.c==="C" && args.sets.d==="D" && JSON.stringify(args.expr)===JSON.stringify(allocation.expr),
          "This design fixture supports the two-entry general, unfamiliar and audit recipe");
        const allocations={}, order=[], pools=[];
        for (const pool of args.budget.pools) {
          const candidates=rows.filter(r=>events.some((e,i)=>e.id===r.id && parent.mask[i]===IN && evaluate(pool.from,args.sets,i)===IN));
          candidates.sort((a,b)=>{
            if (pool.order[0]==="score_desc") return b.score-a.score || a.id.localeCompare(b.id);
            if (pool.order[0]==="seeded_hash") return seededKey(a.id,pool.seed)-seededKey(b.id,pool.seed) || a.id.localeCompare(b.id);
            return a.id.localeCompare(b.id);
          });
          const added=candidates.filter(r=>!allocations[r.id]).slice(0,pool.limit);
          for (const r of added) { allocations[r.id]=pool.name; order.push(r.id); }
          pools.push({name:pool.name,eligible:candidates.length,selected:added.length,unused:pool.limit-added.length});
        }
        const v=publish(["allocate",args.budget],"D",scoped(parent,e=>state(!!allocations[e.id])),
          "Analyst work set","F",{allocations,order,pools});
        return {...summary(v),set:v.handle,capacity:args.budget.total,selected:order.length,ordered_ids:order,
          deferred:parent.counts.in-order.length,pools};
      }
      requireValue(args.mode==="view","Unknown compose mode");
      const region=regions.find(([, ,expr])=>JSON.stringify(expr)===JSON.stringify(args.expr));
      const isD=parent.handle==="U" && region?.[0]==="a_minus_b" && args.sets.a==="A" && args.sets.b==="B";
      const identity=isD ? ["fixture","a_minus_b"] : ["compose",args.scope,args.expr,args.sets];
      const v=publish(identity,
        args.scope,mask,region?.[1] || "Composed view",isD ? "D" : undefined);
      return {...summary(v),view:v.handle};
    }
    function inspectRequest(args) {
      const v=get(args.handle), inside=visible(v.mask);
      switch(args.view) {
      case "overview": return {...summary(v),pools:v.pools || []};
      case "lineage": {
        const path=[];
        for (let ancestor=v; ancestor; ancestor=ancestor.parent ? get(ancestor.parent) : null) {
          path.unshift({handle:ancestor.handle,parent:ancestor.parent,kind:kind(ancestor),title:ancestor.title});
        }
        return {handle:v.handle,snapshot:"U",path};
      }
      case "records": {
        const limit=args.limit ?? 25, prefix=v.handle+"/records/"+JSON.stringify(args.projection ?? "*")+"/";
        requireValue(Number.isInteger(limit) && limit>=1 && limit<=100,"Use a page of 1–100 rows");
        requireValue(!args.cursor || args.cursor.startsWith(prefix),"Cursor belongs to a different view or projection");
        const offset=args.cursor ? Number(args.cursor.slice(prefix.length)) : 0;
        const records=resultRecords(v);
        requireValue(Number.isInteger(offset) && offset>=0 && (!offset || offset<records.length),"Invalid cursor");
        const page=records.slice(offset,offset+limit);
        const projected=args.projection ? page.map(row=>Object.fromEntries(args.projection.map(field=>{
          requireValue(Object.hasOwn(row,field),"Unknown record field"); return [field,row[field]];
        }))) : page;
        return {rows:projected,next_cursor:offset+limit<records.length ? prefix+(offset+limit) : null};
      }
      case "fields": return {field:"host",unit:"occurrences",rows:frequencies(inside,e=>e.host)};
      case "histogram": return {timezone:"UTC",unit:"occurrences",rows:frequencies(inside,e=>bin(e.time))};
      case "regions": {
        const bindings={a:"A",b:"B",c:"C"};
        const counts=regions.map(([key,label,expr,family])=>{
          const subset=scoped(v,(_,i)=>evaluate(expr,bindings,i,true));
          const included=visible(subset);
          return {key,label,family,count:ids(included).length,occurrences:included.length};
        });
        const missingAB=!store.views.has("A") || !store.views.has("B");
        return {unit:"distinct_originals",rows:counts,ab_unclassified:missingAB ? ids(inside).length : 0};
      }
      case "evidence":
        return {...record(v,args.record),reason:!store.views.has("C") ? "Recommendations have not run." :
          opinions[args.record]?.[1] || "Outside the recommendation scope.",
          locations:inside.filter(e=>e.id===args.record).map(({host,time,locator})=>({host,time,locator}))};
      default: throw new Error("Unsupported inspection view");
      }
    }
    function dispatch(tool,args) {
      requireValue(args.v===1,"Expected contract version 1");
      if (tool==="triager_open") {
        if (args.receipt) {
          const handle=store.receipts.get(args.receipt);
          requireValue(handle,"Unknown receipt");
          return {snapshot:"U",receipt:args.receipt,handle,kind:kind(get(handle))};
        }
        requireValue(args.source?.ref==="auth-fixture","Only the design fixture is available");
        return {snapshot:publish(["open"],null,all,"Archive scope","U").handle};
      }
      if (tool==="triager_select") return select(args);
      if (tool==="triager_compose") return compose(args);
      if (tool==="triager_inspect") return inspectRequest(args);
      if (tool==="triager_refine") {
        const parent=get(args.set), {kind,value}=args.action;
        requireValue(["host","hour"].includes(kind),"Unknown refinement");
        const mask=scoped(parent,e=>state((kind==="host" ? e.host : bin(e.time))===value));
        const v=publish(["refine",args.set,kind,value],args.set,mask,kind+" · "+value);
        return {...summary(v),view:v.handle};
      }
      if (tool==="triager_save") {
        const v=get(args.handle);
        requireValue(args.idempotency_key,"Missing save key");
        const receipt="receipt/"+args.idempotency_key;
        requireValue(!store.receipts.has(receipt) || store.receipts.get(receipt)===v.handle,"Save key conflicts with an existing receipt");
        store.receipts.set(receipt,v.handle);
        return {receipt,handle:v.handle,snapshot:"U",kind:kind(v)};
      }
      throw new Error("Unsupported fixture operation: "+tool);
    }
    function call(tool,args) {
      const result=copy({v:1,status:"ready",...dispatch(tool,args)});
      trace.push(copy({tool,args,result}));
      return result;
    }
    const send=(tool,args)=>call(tool,{v:1,...args});
    const read=(handle,view,extra={})=>send("triager_inspect",{handle,view,...extra});
    function inspect(handle) {
      return {...read(handle,"overview"),records:read(handle,"records").rows,hosts:read(handle,"fields").rows,
        bins:read(handle,"histogram").rows,regions:read(handle,"regions").rows};
    }
    function drill(handle,kind,value) {
      if (kind==="region") {
        const r=regions.find(([k])=>k===value);
        requireValue(r,"Unknown region");
        return send("triager_compose",{mode:"view",snapshot:"U",scope:handle,sets:{a:"A",b:"B",c:"C"},expr:r[2]}).view;
      }
      if (kind==="literal" || kind==="glob") {
        return send("triager_select",{scope:handle,producer:{kind:"filter",syntax:kind,query:value}}).set;
      }
      return send("triager_refine",{set:handle,action:{kind,value}}).view;
    }
    return {store,trace,call,inspect,drill,start() {
      for (const [tool,args] of seedArgs) send(tool,args);
      return {snapshot:"U",a:"A",b:"B",difference:"D",c:"C"};
    },evidence:(handle,recordId)=>read(handle,"evidence",{record:recordId}),
    allocate:()=>send("triager_compose",allocation).set,
    save:handle=>send("triager_save",{handle,idempotency_key:"fixture-"+handle}).receipt,
    reopen:receipt=>send("triager_open",{receipt}).handle};
  }
  return {create};
})();

if (typeof window!=="undefined") window.TriagerDemo=TriagerDemo;
if (typeof module!=="undefined" && module.exports) {
  module.exports=TriagerDemo;
  if (require.main===module) {
    const assert=require("node:assert/strict");
    const scenario=require("./walkthrough.json");
    let client=TriagerDemo.create();
    for (const step of scenario.steps) {
      if (step.fresh_client) client=TriagerDemo.create(client.store);
      const result=client.call(step.tool,step.args);
      for (const [path,value] of Object.entries(step.expect)) {
        assert.deepEqual(path.split(".").reduce((o,k)=>o[k],result),value,step.name+" / "+path);
      }
      console.log("PASS  "+step.name);
    }
  }
}
