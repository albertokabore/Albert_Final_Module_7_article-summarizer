"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[6180],{15136:(e,t,o)=>{o.r(t);o.d(t,{main:()=>H});var r=o(94725);var n=o(20979);var s=o(25313);var l=o(56104);var a=o(11114);var i=o(72508);var c=o(2129);var u=o(24911);var p=o(36672);var f=o(1904);var _=o(87779);var A=o(13067);var y=o(67374);var h=o(20135);var d=o(61689);var b=o(34072);var x=o(54336);var j=o(19457);var m=o(43017);var v=o(45695);var g=o(53640);var w=o(367);var C=o(68149);var P=o(87456);var k=o(4380);var E=o(61132);var S=o(57996);var O=o(41884);var N=o(51874);var R=o(90288);var J=o(87145);var L=o(90167);var Q=o(98547);var B=o(57292);var I=o(80046);var M=o(54289);var T=o(40779);var U=o(48552);var Y=o(40005);var z=o(70558);var G=o(31747);var K=o(95527);var V=o(50277);var q=o(77767);var D=o(54549);var F=o(76420);async function $(e,t){try{const o=await window._JUPYTERLAB[e].get(t);const r=o();r.__scope__=e;return r}catch(o){console.warn(`Failed to create module: package: ${e}; module: ${t}`);throw o}}async function H(){var e=r.PageConfig.getOption("browserTest");if(e.toLowerCase()==="true"){var t=document.createElement("div");t.id="browserTest";document.body.appendChild(t);t.textContent="[]";t.style.display="none";var n=[];var s=false;var l=25e3;var a=function(){if(s){return}s=true;t.className="completed"};window.onerror=function(e,o,r,s,l){n.push(String(l));t.textContent=JSON.stringify(n)};console.error=function(e){n.push(String(e));t.textContent=JSON.stringify(n)}}var i=o(6323).JupyterLab;var c=[];var u=[];var p=[];var f=[];const _=[];const A=[];const y=[];const h=JSON.parse(r.PageConfig.getOption("federated_extensions"));const d=[];h.forEach((e=>{if(e.extension){d.push(e.name);_.push($(e.name,e.extension))}if(e.mimeExtension){d.push(e.name);A.push($(e.name,e.mimeExtension))}if(e.style&&!r.PageConfig.Extension.isDisabled(e.name)){y.push($(e.name,e.style))}}));const b=[];function*x(e){let t;if(e.hasOwnProperty("__esModule")){t=e.default}else{t=e}let o=Array.isArray(t)?t:[t];for(let n of o){const t=r.PageConfig.Extension.isDisabled(n.id);b.push({id:n.id,description:n.description,requires:n.requires??[],optional:n.optional??[],provides:n.provides??null,autoStart:n.autoStart,enabled:!t,extension:e.__scope__});if(t){c.push(n.id);continue}if(r.PageConfig.Extension.isDeferred(n.id)){u.push(n.id);p.push(n.id)}yield n}}const j=[];if(!d.includes("@jupyterlab/javascript-extension")){try{let e=o(28281);e.__scope__="@jupyterlab/javascript-extension";for(let t of x(e)){j.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/json-extension")){try{let e=o(86853);e.__scope__="@jupyterlab/json-extension";for(let t of x(e)){j.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/mermaid-extension")){try{let e=o(47375);e.__scope__="@jupyterlab/mermaid-extension";for(let t of x(e)){j.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/pdf-extension")){try{let e=o(70135);e.__scope__="@jupyterlab/pdf-extension";for(let t of x(e)){j.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/vega5-extension")){try{let e=o(68475);e.__scope__="@jupyterlab/vega5-extension";for(let t of x(e)){j.push(t)}}catch(P){console.error(P)}}const m=await Promise.allSettled(A);m.forEach((e=>{if(e.status==="fulfilled"){for(let t of x(e.value)){j.push(t)}}else{console.error(e.reason)}}));if(!d.includes("@jupyterlab/application-extension")){try{let e=o(16535);e.__scope__="@jupyterlab/application-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/apputils-extension")){try{let e=o(23733);e.__scope__="@jupyterlab/apputils-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/cell-toolbar-extension")){try{let e=o(76061);e.__scope__="@jupyterlab/cell-toolbar-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/celltags-extension")){try{let e=o(11349);e.__scope__="@jupyterlab/celltags-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/codemirror-extension")){try{let e=o(4889);e.__scope__="@jupyterlab/codemirror-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/completer-extension")){try{let e=o(857);e.__scope__="@jupyterlab/completer-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/console-extension")){try{let e=o(70121);e.__scope__="@jupyterlab/console-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/csvviewer-extension")){try{let e=o(86659);e.__scope__="@jupyterlab/csvviewer-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/debugger-extension")){try{let e=o(54713);e.__scope__="@jupyterlab/debugger-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/docmanager-extension")){try{let e=o(84005);e.__scope__="@jupyterlab/docmanager-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/documentsearch-extension")){try{let e=o(58841);e.__scope__="@jupyterlab/documentsearch-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/extensionmanager-extension")){try{let e=o(349);e.__scope__="@jupyterlab/extensionmanager-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/filebrowser-extension")){try{let e=o(98815);e.__scope__="@jupyterlab/filebrowser-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/fileeditor-extension")){try{let e=o(29289);e.__scope__="@jupyterlab/fileeditor-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/help-extension")){try{let e=o(68929);e.__scope__="@jupyterlab/help-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/htmlviewer-extension")){try{let e=o(2441);e.__scope__="@jupyterlab/htmlviewer-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/hub-extension")){try{let e=o(74201);e.__scope__="@jupyterlab/hub-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/imageviewer-extension")){try{let e=o(89841);e.__scope__="@jupyterlab/imageviewer-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/inspector-extension")){try{let e=o(79033);e.__scope__="@jupyterlab/inspector-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/launcher-extension")){try{let e=o(11633);e.__scope__="@jupyterlab/launcher-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/logconsole-extension")){try{let e=o(74745);e.__scope__="@jupyterlab/logconsole-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/lsp-extension")){try{let e=o(25337);e.__scope__="@jupyterlab/lsp-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/mainmenu-extension")){try{let e=o(39209);e.__scope__="@jupyterlab/mainmenu-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/markdownviewer-extension")){try{let e=o(59865);e.__scope__="@jupyterlab/markdownviewer-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/markedparser-extension")){try{let e=o(44945);e.__scope__="@jupyterlab/markedparser-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/mathjax-extension")){try{let e=o(16881);e.__scope__="@jupyterlab/mathjax-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/mermaid-extension")){try{let e=o(87609);e.__scope__="@jupyterlab/mermaid-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/metadataform-extension")){try{let e=o(53281);e.__scope__="@jupyterlab/metadataform-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/notebook-extension")){try{let e=o(98057);e.__scope__="@jupyterlab/notebook-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/pluginmanager-extension")){try{let e=o(69111);e.__scope__="@jupyterlab/pluginmanager-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/rendermime-extension")){try{let e=o(21597);e.__scope__="@jupyterlab/rendermime-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/running-extension")){try{let e=o(64409);e.__scope__="@jupyterlab/running-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/settingeditor-extension")){try{let e=o(42425);e.__scope__="@jupyterlab/settingeditor-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/shortcuts-extension")){try{let e=o(25816);e.__scope__="@jupyterlab/shortcuts-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/statusbar-extension")){try{let e=o(1865);e.__scope__="@jupyterlab/statusbar-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/terminal-extension")){try{let e=o(24493);e.__scope__="@jupyterlab/terminal-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/theme-dark-extension")){try{let e=o(10125);e.__scope__="@jupyterlab/theme-dark-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/theme-dark-high-contrast-extension")){try{let e=o(97581);e.__scope__="@jupyterlab/theme-dark-high-contrast-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/theme-light-extension")){try{let e=o(45819);e.__scope__="@jupyterlab/theme-light-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/toc-extension")){try{let e=o(37405);e.__scope__="@jupyterlab/toc-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/tooltip-extension")){try{let e=o(38377);e.__scope__="@jupyterlab/tooltip-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/translation-extension")){try{let e=o(56073);e.__scope__="@jupyterlab/translation-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/ui-components-extension")){try{let e=o(66165);e.__scope__="@jupyterlab/ui-components-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}if(!d.includes("@jupyterlab/workspaces-extension")){try{let e=o(23785);e.__scope__="@jupyterlab/workspaces-extension";for(let t of x(e)){f.push(t)}}catch(P){console.error(P)}}const v=await Promise.allSettled(_);v.forEach((e=>{if(e.status==="fulfilled"){for(let t of x(e.value)){f.push(t)}}else{console.error(e.reason)}}));(await Promise.allSettled(y)).filter((({status:e})=>e==="rejected")).forEach((({reason:e})=>{console.error(e)}));const g=new i({mimeExtensions:j,disabled:{matches:c,patterns:r.PageConfig.Extension.disabled.map((function(e){return e.raw}))},deferred:{matches:u,patterns:r.PageConfig.Extension.deferred.map((function(e){return e.raw}))},availablePlugins:b});f.forEach((function(e){g.registerPluginModule(e)}));g.start({ignorePlugins:p,bubblingKeydown:true});var w=(r.PageConfig.getOption("exposeAppInBrowser")||"").toLowerCase()==="true";var C=(r.PageConfig.getOption("devMode")||"").toLowerCase()==="true";if(w||C){window.jupyterapp=g}if(e.toLowerCase()==="true"){g.restored.then((function(){a(n)})).catch((function(e){a([`RestoreError: ${e.message}`])}));window.setTimeout((function(){a(n)}),l)}}},78269:e=>{e.exports="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII="}}]);