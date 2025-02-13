<script lang="ts">
	import { FileDropzone } from '@skeletonlabs/skeleton';
	import { toastStore } from '@skeletonlabs/skeleton';
	import type { ToastSettings } from '@skeletonlabs/skeleton';
	import { onMount } from 'svelte';
	import DownloadJson from '../../../components/download-json.svelte';

	let files: FileList;
	let uploaded: string;
	let downloaded: string;
	let hl: string;
	let hh: string;
	let sl: string;
	let sh: string;
	let vl: string;
	let vh: string;
	let fill: string;
	let colored: boolean = true;
	let invert: boolean = false;
	let imageObj: any;
	let canvas_width = 0;
	let canvas_height = 0;

	function rgb2hsv(r, g, b) {
		let rabs, gabs, babs, rr, gg, bb, h, s, v, diff, diffc, percentRoundFn;
		rabs = r / 255;
		gabs = g / 255;
		babs = b / 255;
		(v = Math.max(rabs, gabs, babs)), (diff = v - Math.min(rabs, gabs, babs));
		diffc = (c) => (v - c) / 6 / diff + 1 / 2;
		percentRoundFn = (num) => Math.round(num * 100) / 100;
		if (diff == 0) {
			h = s = 0;
		} else {
			s = diff / v;
			rr = diffc(rabs);
			gg = diffc(gabs);
			bb = diffc(babs);

			if (rabs === v) {
				h = bb - gg;
			} else if (gabs === v) {
				h = 1 / 3 + rr - bb;
			} else if (babs === v) {
				h = 2 / 3 + gg - rr;
			}
			if (h < 0) {
				h += 1;
			} else if (h > 1) {
				h -= 1;
			}
		}

		let h_res = Math.round(h * 180);
		let s_res = percentRoundFn(s * 255);
		let v_res = percentRoundFn(v * 255);

		return { h: h_res, s: s_res, v: v_res };
	}

	onMount(() => {
		imageObj = new Image();

		let d = document.getElementById('ah');
		if (d) {
			canvas_width = d.offsetWidth;
			canvas_height = d.offsetHeight - 100;
		}
		// alert(canvas_height);
	});

	interface Preset {
		name: string;
		hl: string;
		hh: string;
		sl: string;
		sh: string;
		vl: string;
		vh: string;
		fill: string;
		colored: boolean;
		invert: boolean;
	}

	let presets: Array<Preset> = [];

	presets.push({
		name: 'green',
		hl: '20',
		hh: '80',
		sl: '10',
		sh: '255',
		vl: '125',
		vh: '255',
		fill: '50',
		colored: true,
		invert: false
	});

	presets.push({
		name: 'blue',
		hl: '50',
		hh: '140',
		sl: '0',
		sh: '255',
		vl: '0',
		vh: '255',
		fill: '50',
		colored: true,
		invert: true
	});

	function activatePreset(p: Preset) {
		// console.log(p.name);
		for (const property in p) {
			if (property != 'name') {
				// let e = document.getElementById(property);
				// if (e) {
				// 	e.value = p[property];
				// }
				switch (property) {
					case 'hl':
						hl = p[property];
						break;

					case 'hh':
						hh = p[property];
						break;

					case 'sl':
						sl = p[property];
						break;

					case 'sh':
						sh = p[property];
						break;

					case 'vl':
						vl = p[property];
						break;

					case 'vh':
						vh = p[property];
						break;

					case 'fill':
						fill = p[property];
						break;

					case 'colored':
						colored = p[property];
						break;

					case 'invert':
						invert = p[property];
						break;
					default:
						break;
				}
			}
		}
	}

	function onChangeHandler(e: Event): void {
		// console.log('file data:', e);
		// console.log('test:', files);
		let reader = new FileReader();
		reader.readAsDataURL(files[0]);
		reader.onload = (e) => {
			if (e.target) {
				if (typeof e.target.result == 'string') {
					uploaded = e.target.result;

					if (imageObj) {
						let canvas = document.getElementById('canvas');
						if (canvas) {
							let ctx = canvas.getContext('2d');

							imageObj.onload = function () {
								// ctx.canvas.width = window.innerWidth;
								// ctx.canvas.height = window.innerHeight;
								let image_width = imageObj.width;
								let image_height = imageObj.height;

								let hRatio = canvas_width / image_width;
								let vRatio = canvas_height / image_height;

								let ratio = Math.min(hRatio, vRatio);

								ctx.canvas.width = canvas_width;
								ctx.canvas.height = canvas_height;
								ctx.drawImage(imageObj, 0, 0, image_width * ratio, image_height * ratio);
							};

							imageObj.src = uploaded;

							const hoveredColor = document.getElementById('hovered-color');
							const selectedColor = document.getElementById('selected-color');
							const hovered_color_text_h = document.getElementById('hovered-color-text-h');
							const hovered_color_text_s = document.getElementById('hovered-color-text-s');
							const hovered_color_text_v = document.getElementById('hovered-color-text-v');

							const selected_color_text_h = document.getElementById('selected-color-text-h');
							const selected_color_text_s = document.getElementById('selected-color-text-s');
							const selected_color_text_v = document.getElementById('selected-color-text-v');

							function pick(event: MouseEvent, destination: HTMLElement) {
								if (canvas) {
									const bounding = canvas.getBoundingClientRect();
									const x = event.clientX - bounding.left;
									const y = event.clientY - bounding.top;
									const pixel = ctx.getImageData(x, y, 1, 1);
									const data = pixel.data;

									const rgba = `rgba(${data[0]}, ${data[1]}, ${data[2]}, ${data[3] / 255})`;
									destination.style.background = rgba;
									// destination.textContent = rgb2hsv(data[0], data[1], data[2]);

									// return rgba;
									let res = rgb2hsv(data[0], data[1], data[2]);
									return res;
								}
							}
							if (
								hoveredColor &&
								selectedColor &&
								hovered_color_text_h &&
								hovered_color_text_s &&
								hovered_color_text_v &&
								selected_color_text_h &&
								selected_color_text_s &&
								selected_color_text_v
							) {
								canvas.addEventListener('mousemove', (event) => {
									let rgb = pick(event, hoveredColor);
									if (rgb) {
										hovered_color_text_h.textContent = String(rgb.h);
										hovered_color_text_s.textContent = String(rgb.s);
										hovered_color_text_v.textContent = String(rgb.v);
									}
								});
								canvas.addEventListener('click', (event) => {
									// alert("click");
									let rgb = pick(event, selectedColor);
									if (rgb) {
										selected_color_text_h.textContent = String(rgb.h);
										selected_color_text_s.textContent = String(rgb.s);
										selected_color_text_v.textContent = String(rgb.v);
									}
								});
							}
						}
					}
				}
			}
		};
	}

	async function preview() {
		if (uploaded) {
			const t: ToastSettings = {
				message: 'Sending...',
				background: 'variant-filled-primary',
				timeout: 5000
			};
			toastStore.trigger(t);
			// send request
			console.log(uploaded);
			// console.log("fx: "+fx);
			let fetchResponse = await fetch('http://localhost:8000/threshold', {
				method: 'POST',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					img: uploaded,
					hl: hl,
					hh: hh,
					sl: sl,
					sh: sh,
					vl: vl,
					vh: vh,
					fill: fill,
					colored: colored,
					invert: invert
				})
			});

			let data = await fetchResponse.json();
			console.log(data);
			let code = fetchResponse.status;
			console.log(code);
			if (code === 200) {
				downloaded = data;
				const t: ToastSettings = {
					message: 'Success',
					background: 'variant-filled-success',
					timeout: 5000
				};
				toastStore.trigger(t);
			} else {
				const t: ToastSettings = {
					message: 'Something went wrong',
					background: 'variant-filled-error',
					timeout: 5000
				};
				toastStore.trigger(t);
			}
		} else {
			const t: ToastSettings = {
				message: 'Upload an image first!',
				background: 'variant-filled-error',
				timeout: 5000
			};
			toastStore.trigger(t);
		}
	}

	async function histogram() {
		if (uploaded) {
			const t: ToastSettings = {
				message: 'Sending...',
				background: 'variant-filled-primary',
				timeout: 5000
			};
			toastStore.trigger(t);
			// send request
			console.log(uploaded);
			// console.log("fx: "+fx);
			let fetchResponse = await fetch('http://localhost:8000/histogram', {
				method: 'POST',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					img: uploaded
				})
			});

			let data = await fetchResponse.json();
			console.log(data);
			let code = fetchResponse.status;
			console.log(code);
			if (code === 200) {
				downloaded = data;
				const t: ToastSettings = {
					message: 'Success',
					background: 'variant-filled-success',
					timeout: 5000
				};
				toastStore.trigger(t);
			} else {
				const t: ToastSettings = {
					message: 'Something went wrong',
					background: 'variant-filled-error',
					timeout: 5000
				};
				toastStore.trigger(t);
			}
		} else {
			const t: ToastSettings = {
				message: 'Upload an image first!',
				background: 'variant-filled-error',
				timeout: 5000
			};
			toastStore.trigger(t);
		}
	}
</script>

<div class="">
	<div class="grid grid-rows-1 grid-cols-3 gap-5">
		<div class="max-h-screen" id="ah">
			<div id="thediv" class="">
				<div class="grid grid-cols-3 grid-rows-3">
					<p />
					<div class="grid grid-cols-3 grid-rows-1">
						<p>H</p>
						<p>S</p>
						<p>V</p>
					</div>
					<p />
					<p>Hovering:</p>
					<div class="grid grid-cols-3 grid-rows-1">
						<p id="hovered-color-text-h" />
						<p id="hovered-color-text-s" />
						<p id="hovered-color-text-v" />
					</div>
					<!-- <p id="hovered-color-text"></p> -->
					<td class="color-cell w-full" id="hovered-color" />

					<p>Clicked:</p>
					<div class="grid grid-cols-3 grid-rows-1">
						<p id="selected-color-text-h" />
						<p id="selected-color-text-s" />
						<p id="selected-color-text-v" />
					</div>
					<!-- <p id="selected-color-text"></p> -->
					<td class="color-cell w-full" id="selected-color" />
				</div>

				<canvas id="canvas" class="" />
				<!-- <FileDropzone class="mt-5" name="files" bind:files on:change={onChangeHandler} /> -->
				<FileDropzone class="mt-5" name="files" bind:files on:change={onChangeHandler} />
			</div>
		</div>

		<div class="">
			<div class="grid grid-rows-4 grid-cols-2 pt-5 text-center">
				<h1>Low</h1>
				<h1>High</h1>
				<input bind:value={hl} class="input" type="text" id="hl" name="hl" placeholder="hl" />
				<input bind:value={hh} class="input" type="text" id="hh" name="hh" placeholder="hh" />
				<input bind:value={sl} class="input" type="text" id="sl" name="sl" placeholder="sl" />
				<input bind:value={sh} class="input" type="text" id="sh" name="sh" placeholder="sh" />
				<input bind:value={vl} class="input" type="text" id="vl" name="vl" placeholder="vl" />
				<input bind:value={vh} class="input" type="text" id="vh" name="vh" placeholder="vh" />
			</div>
			<h1>Fill objects smaller than (px):</h1>
			<input bind:value={fill} class="input" type="text" id="fill" name="fill" placeholder="fill" />
			<div class="flex justify-center">
				<img class="pt-4" src="/images/hue_and_sat.png" alt="hsv color model" />
			</div>
			<div class="text-center pt-20">
				<div class="">
					<input bind:checked={colored} class="checkbox" type="checkbox" />
					<p class="pt-2 pb-6">Colored mask?</p>
					<input bind:checked={invert} class="checkbox" type="checkbox" />
					<p class="pt-2 pb-6">Invert mask?</p>
				</div>
				{#if uploaded}
					<button class="btn variant-filled-success" on:click={preview}>Preview</button>
					<button class="btn variant-filled-success" on:click={histogram}>Show histogram</button>
				{:else}
					<button class="btn variant-filled-error" on:click={preview}>Preview</button>
				{/if}
				<DownloadJson
					filename="threshold.json"
					dataObjToWrite={{
						hl: hl,
						hh: hh,
						sl: sl,
						sh: sh,
						vl: vl,
						vh: vh,
						fill: fill,
						invert: invert
					}}
				/>
			</div>
		</div>

		<div class="mt-16">
			{#if downloaded}
				<img id="download-image" alt="Downloaded" src="data:image/png;base64,{downloaded}" />
			{:else}
				<!-- <img
					id="input-image"
					alt="Uploaded"
					src="https://www.opticallimits.com/images/8Reviews/lenses/samyang_8_35_eos/8mm.jpg"
				/> -->
			{/if}
		</div>
	</div>
	<!-- <div class="grid grid-rows-1 grid-cols-3">
		<FileDropzone class="mt-5" name="files" bind:files on:change={onChangeHandler} />
	</div> -->
	<div class="text-center">
		<h1 class="py-6">Presets</h1>
		<div class="flex items-center justify-center h-full">
			<div class="grid grid-cols-1 gap-7">
				{#each presets as preset}
					<button class="btn variant-filled mx-3" on:click={() => activatePreset(preset)}
						>{preset.name}</button
					>
				{/each}
			</div>
		</div>
	</div>
</div>
