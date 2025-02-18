<script lang="ts">
	import { FileDropzone } from '@skeletonlabs/skeleton';
	import { toastStore } from '@skeletonlabs/skeleton';
	import type { ToastSettings } from '@skeletonlabs/skeleton';
	import DownloadJson from '../../../components/download-json.svelte';

	let files: FileList;
	let uploaded: string;
	let downloaded: string;
	let fx: string;
	let fy: string;
	let cx: string;
	let cy: string;
	let k1: string;
	let k2: string;
	let k3: string;
	let k4: string;

	interface Preset {
		name: string;
		fx: string;
		fy: string;
		cx: string;
		cy: string;
		k1: string;
		k2: string;
		k3: string;
		k4: string;
	}

	let presets: Array<Preset> = [];

	presets.push({
		name: 'pi cam',
		fx: '1800',
		fy: '1800',
		cx: '1296',
		cy: '972',
		k1: '0',
		k2: '0',
		k3: '0',
		k4: '0'
	});

	presets.push({
		name: 'another preset',
		fx: 'string',
		fy: 'string',
		cx: 'string',
		cy: 'string',
		k1: '0',
		k2: '0',
		k3: '0',
		k4: '0'
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
					case 'fx':
						fx = p[property];
						break;

					case 'fy':
						fy = p[property];
						break;

					case 'cx':
						cx = p[property];
						break;

					case 'cy':
						cy = p[property];
						break;

					case 'k1':
						k1 = p[property];
						break;

					case 'k2':
						k2 = p[property];
						break;

					case 'k3':
						k3 = p[property];
						break;

					case 'k4':
						k4 = p[property];
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
			let fetchResponse = await fetch('http://localhost:8000/undistort-fisheye', {
				method: 'POST',
				headers: {
					Accept: 'application/json',
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					img: uploaded,
					fx: fx,
					fy: fy,
					cx: cx,
					cy: cy,
					k1: k1,
					k2: k2,
					k3: k3,
					k4: k4
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
		<div class="">
			{#if uploaded}
				<img id="input-image" alt="Uploaded" src={uploaded} />
			{:else}
				<!-- <img
					id="input-image"
					alt="Uploaded"
					src="https://www.opticallimits.com/images/8Reviews/lenses/samyang_8_35_eos/8mm.jpg"
				/> -->
			{/if}
		</div>

		<div class="">
			<div class="grid grid-rows-1 grid-cols-4 pt-5">
				<div class="text-right pr-5">
					<h1>K =</h1>
				</div>

				<div class="col-span-2">
					<div class="grid grid-cols-3 grid-rows-3 gap-y-3 gap-x-3 text-center">
						<input bind:value={fx} class="input" type="text" id="fx" name="fx" placeholder="fx" />
						<h1>0</h1>
						<input bind:value={cx} class="input" type="text" id="cx" name="cx" placeholder="cx" />

						<h1>0</h1>
						<input bind:value={fy} class="input" type="text" id="fy" name="fy" placeholder="fy" />
						<input bind:value={cy} class="input" type="text" id="cy" name="cy" placeholder="cy" />

						<h1>0</h1>
						<h1>0</h1>
						<h1>1</h1>
					</div>
				</div>
			</div>
			<div class="grid grid-rows-1 grid-cols-4 pt-20">
				<div class="text-right pr-5">
					<h1>D =</h1>
				</div>

				<div class="col-span-2">
					<div class="grid grid-cols-4 grid-rows-1 gap-x-3">
						<input bind:value={k1} class="input" type="text" id="k1" name="k1" placeholder="k1" />
						<input bind:value={k2} class="input" type="text" id="k2" name="k2" placeholder="k2" />
						<input bind:value={k3} class="input" type="text" id="k3" name="k3" placeholder="k3" />
						<input bind:value={k4} class="input" type="text" id="k4" name="k4" placeholder="k4" />
					</div>
				</div>
			</div>
			<div class="text-center pt-20">
				{#if uploaded}
					<button class="btn variant-filled-success" on:click={preview}>Preview</button>
				{:else}
					<button class="btn variant-filled-error" on:click={preview}>Preview</button>
				{/if}
				<DownloadJson
					filename="undistort.json"
					dataObjToWrite={{
						fx: fx,
						fy: fy,
						cx: cx,
						cy: cy,
						k1: k1,
						k2: k2,
						k3: k3,
						k4: k4
					}}
				/>
			</div>
		</div>

		<div class="">
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
	<div class="grid grid-rows-1 grid-cols-3">
		<FileDropzone class="mt-5" name="files" bind:files on:change={onChangeHandler} />
	</div>
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
