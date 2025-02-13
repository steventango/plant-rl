<script lang="ts">
	// The ordering of these imports is critical to your app working properly
	import '@skeletonlabs/skeleton/themes/theme-seafoam.css';
	// If you have source.organizeImports set to true in VSCode, then it will auto change this ordering
	import '@skeletonlabs/skeleton/styles/skeleton.css';
	// Most of your app wide CSS should be put in this file
	import '../app.postcss';
	import { AppShell, AppBar, LightSwitch } from '@skeletonlabs/skeleton';
	import { Toast, toastStore } from '@skeletonlabs/skeleton';
	import type { ToastSettings } from '@skeletonlabs/skeleton';
	import { Modal, modalStore } from '@skeletonlabs/skeleton';
	import type { ModalSettings, ModalComponent } from '@skeletonlabs/skeleton';
	import { page } from '$app/stores';

	async function fetchWithTimeout(resource: string, options: any = {}) {
		const { timeout = 8000 } = options;

		const controller = new AbortController();
		const id = setTimeout(() => controller.abort(), timeout);

		const response = await fetch(resource, {
			...options,
			signal: controller.signal
		});
		clearTimeout(id);

		return response;
	}

	async function ping() {
		const t: ToastSettings = {
			message: 'Sending ping...',
			background: 'variant-filled-primary',
			timeout: 5000
		};
		toastStore.trigger(t);
		// send request
		try {
			let fetchResponse = await fetchWithTimeout('http://localhost:8000/ping', { timeout: 6000 });

			let data = await fetchResponse.json();
			let code = fetchResponse.status;
			if (code === 200) {
				const t: ToastSettings = {
					message: 'Server is UP',
					background: 'variant-filled-success',
					timeout: 5000
				};
				toastStore.trigger(t);
			}
		} catch (error) {
			// console.log(error);
			const t: ToastSettings = {
				message: 'Server is DOWN',
				background: 'variant-filled-error',
				timeout: 5000
			};
			toastStore.trigger(t);
		}
	}
</script>
<Modal></Modal>

<!-- App Shell -->
<AppShell>
	<svelte:fragment slot="header">
		<!-- App Bar -->
		<AppBar background="bg-transparent">
			<svelte:fragment slot="lead">
				<a href="/" class="text-xl uppercase">🌱 Uhrig Lab Tools</a>
			</svelte:fragment>
			<svelte:fragment slot="trail">
				<button class="btn btn-sm variant-ghost-surface" on:click={ping}>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="1.41em"
						height="1em"
						viewBox="0 0 1984 1408"
						><path
							fill="currentColor"
							d="M992 1395q-20 0-93-73.5t-73-93.5q0-32 62.5-54t103.5-22t103.5 22t62.5 54q0 20-73 93.5t-93 73.5zm270-271q-2 0-40-25t-101.5-50t-128.5-25t-128.5 25t-101 50t-40.5 25q-18 0-93.5-75T553 956q0-13 10-23q78-77 196-121t233-44t233 44t196 121q10 10 10 23q0 18-75.5 93t-93.5 75zm273-272q-11 0-23-8q-136-105-252-154.5T992 640q-85 0-170.5 22t-149 53T559 777t-79 53t-31 22q-17 0-92-75t-75-93q0-12 10-22q132-132 320-205t380-73t380 73t320 205q10 10 10 22q0 18-75 93t-92 75zm271-271q-11 0-22-9q-179-157-371.5-236.5T992 256t-420.5 79.5T200 572q-11 9-22 9q-17 0-92.5-75T10 413q0-13 10-23q187-186 445-288T992 0t527 102t445 288q10 10 10 23q0 18-75.5 93t-92.5 75z"
						/></svg
					>
				</button>

				<button on:click={() => { 
					console.log($page.url.pathname);
					switch($page.url.pathname) {
						case "/tools/inspect": {
							console.log("oi");
							break;
						}
						default: {
							console.log("default");
						}
					}
					
					modalStore.trigger({
					buttonTextCancel: 'Close',
					type: 'alert',
					title: 'Example Alert',
					body: 'This is an example modal.',
					image: 'https://i.imgur.com/WOgTG96.gif',
				});
				
				}} class="btn btn-sm variant-ghost-surface"> Help </button>
				<!-- <a
					class="btn btn-sm variant-ghost-surface"
					href="https://twitter.com/SkeletonUI"
					target="_blank"
					rel="noreferrer"
				>
					Twitter
				</a> -->
				<!-- <a
					class="btn btn-sm variant-ghost-surface"
					href="https://github.com/skeletonlabs/skeleton"
					target="_blank"
					rel="noreferrer"
				>
					GitHub
				</a> -->
				<LightSwitch />
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>
	<svelte:fragment slot="footer">
		<Toast />
	</svelte:fragment>
	<!-- Page Route Content -->
	<slot />
</AppShell>
