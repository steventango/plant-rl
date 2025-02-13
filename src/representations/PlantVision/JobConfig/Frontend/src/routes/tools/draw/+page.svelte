<!--
https://eugenkiss.github.io/7guis/tasks#circle

Click on the canvas to draw a circle. Click on a circle
to select it. Right-click on the canvas to adjust the
radius of the selected circle.
-->

<script lang="ts">
	// import { SlideToggle } from "@skeletonlabs/skeleton";
	// import { toastStore } from '@skeletonlabs/skeleton';
	import DownloadJson from '../../../components/download-json.svelte';
	import { onMount } from 'svelte';
	import { FileButton } from '@skeletonlabs/skeleton';

	// const t = {
	// 			message: 'Sending...',
	// 			background: 'variant-filled-primary',
	// 			timeout: 5000
	// 		};
	// toastStore.trigger(t);

	// let i = 0;
	// let undoStack = [[]];
	let circles = [];
	let fixedcircles = [];
	let selected;
	let adjusting = false;
	let adjusted = false;
	let lastr = 50;
	let lastgenotype = 'genotype';
	let i = 0;
	let files: FileList;
	// let ordering = false;

	function clear() {
		// // just for testing
		// // need to adjust rois based on these dimensions
		// console.log(document.getElementById("image")?.clientWidth)
		// console.log(document.getElementById("image")?.clientHeight)

		// // and
		// let myImg = document.querySelector("#image");
		// let realWidth = myImg.naturalWidth;
		// let realHeight = myImg.naturalHeight;
		// console.log(realWidth);
		// console.log(realHeight);

		// undoStack = [[]];
		circles = [];
		i = 0;
		adjusting = false;
		adjusted = false;
		lastr = 50;
		lastgenotype = 'genotype';
		// ordering = false;
	}
	function fixROIS(circles) {
			console.log("hi");
			let clientWidth = document.getElementById('image')?.clientWidth;
			let clientHeight = document.getElementById('image')?.clientHeight;

			let realWidth = document.querySelector('#image')?.naturalWidth;
			let realHeight = document.querySelector('#image')?.naturalHeight;

			// let myImg = document.querySelector('#image');
			// if (myImg) {
			// let realWidth = myImg.naturalWidth;
			// let realHeight = myImg.naturalHeight;
			// }
			// console.log(realWidth);
			// console.log(realHeight);
			
			if (clientHeight && clientWidth && realWidth && realHeight) {
				let widthRatio = realWidth / clientWidth;
				let heightRatio = realHeight / clientHeight;
				fixedcircles = [];
				circles.forEach((circle) => {
					fixedcircles.push({
						cx: circle.cx * widthRatio,
						cy: circle.cy * heightRatio,
						r: circle.r * widthRatio,
						number: circle.number,
						genotype: circle.genotype
						// order: i,
					});
				});
			}
			console.log("bye");
			console.log(fixedcircles);
			return fixedcircles;
		}

	// function order() {
	// 	ordering = true;
	// }

	function handleClick(event) {
		var bounds = event.target.getBoundingClientRect();
		if (adjusting) {
			adjusting = false;

			// if circle was adjusted,
			// push to the stack
			if (adjusted) push();
			return;
		}

		// if (!ordering) {
		// 	i = 0;
		// }

		const circle = {
			cx: event.pageX - bounds.left - scrollX,
			cy: event.pageY - bounds.top - scrollY,
			r: lastr,
			number: i++,
			genotype: lastgenotype
			// order: i,
		};
		// console.log("placed circle #"+circle.order);

		circles = circles.concat(circle); // can i just use push here?
		selected = circle;

		push();
	}

	function adjust(event) {
		selected.r = +event.target.value;
		lastr = selected.r;
		circles = circles;
		adjusted = true;
	}

	function select(circle, event) {
		if (selected === circle) {
			// console.log("delete");
			i = i - 1;
			circles = circles.filter(function (value, index, arr) {
				return value !== circle;
			});
			// handle undostack
		}
		if (!adjusting) {
			event.stopPropagation();
			selected = circle;
		}
	}

	function push() {
		// const newUndoStack = undoStack.slice(0, ++i);
		// newUndoStack.push(clone(circles));
		// undoStack = newUndoStack;
	}

	function travel(d) {
		// circles = clone(undoStack[i += d]);
		adjusting = false;
	}

	function clone(circles) {
		return circles.map(({ cx, cy, r }) => ({ cx, cy, r }));
	}

	function onChangeHandler(e: Event): void {
	console.log('file data:', e);
	let reader = new FileReader();
	reader.onload = function () {
            document.getElementById("image").src = reader.result;
        }
        reader.readAsDataURL(files[0]);
}
</script>

<div class="controls">
	<h1>Remember to upload an undistorted image otherwise your ROI's will shift!</h1>
	<!-- <button class="btn btn-sm variant-ghost-surface" on:click="{() => travel(-1)}" disabled="{i === 0}">undo</button>
	<button class="btn btn-sm variant-ghost-surface" on:click="{() => travel(+1)}" disabled="{i === undoStack.length -1}">redo</button> -->
	<button class="btn btn-sm variant-ghost-surface" on:click={clear}>clear</button>
		<DownloadJson onclickprep={fixROIS} onclickpreparg={circles} filename="rois.json"/>
		<FileButton class="mt-5" name="files" bind:files={files} on:change={onChangeHandler} />
	<!-- <SlideToggle name="slider-label" bind:checked={ordering}>Set ordering</SlideToggle> -->
	<!-- <button class="btn btn-sm variant-ghost-surface" on:click="{order}">add ordering</button> -->
</div>

<div class="relative">
	<img
		id="image"
		class="block max-w-full h-auto"
		src={files}
		alt="Drawing surface"
	/>
	<svg class="absolute top-0 left-0" on:click={handleClick} on:keydown={handleClick}>
		{#each circles as circle}
			<circle
				cx={circle.cx}
				cy={circle.cy}
				r={circle.r}
				on:click={(event) => select(circle, event)}
				on:keydown={(event) => select(circle, event)}
				on:contextmenu|stopPropagation|preventDefault={() => {
					adjusting = !adjusting;
					if (adjusting) selected = circle;
				}}
				fill={circle === selected ? '#ff00d930' : '#00000000'}
			/>
		{/each}
	</svg>
</div>

{#if adjusting}
	<div class="adjuster">
		<p class="text-black">adjust diameter of circle at {selected.cx}, {selected.cy}</p>
		<!-- <input type="range" value={selected.r} on:input={adjust}> -->
		<input type="text" value={selected.r} on:input={adjust} />
		<br />
		<input type="text" bind:value={selected.number} />
		<br />
		<input
			type="text"
			bind:value={selected.genotype}
			on:input={() => {
				lastgenotype = selected.genotype;
			}}
		/>
	</div>
{/if}

<style>
	.controls {
		/* position: absolute; */
		width: 100%;
		text-align: center;
	}

	svg {
		background-color: #00000000;
		width: 100%;
		height: 100%;
		/* background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 96 43'><path class='color' d='M 55.584673,43.175632 36.75281,22.967243 0,43.175632 40.42403,0 59.71135,20.208389 96,0 Z' /></svg>"); */
	}

	circle {
		stroke: rgb(255, 0, 0);
		stroke-width: 3pt;
	}

	.adjuster {
		position: absolute;
		width: 80%;
		top: 70%;
		left: 50%;
		transform: translate(-50%, -50%);
		padding: 1em;
		text-align: center;
		background-color: rgba(255, 255, 255, 0.5);
		border-radius: 4px;
	}

	/* input[type='range'] {
		width: 100%;
	} */
</style>
