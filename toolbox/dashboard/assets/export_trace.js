(function () {
	"use strict";

	const BUTTON_CLASS = "download-trace-json";

	function asSerializableArray(values) {
		if (values === undefined || values === null) {
			return [];
		}

		return Array.from(values, value => {
			if (Array.isArray(value) || ArrayBuffer.isView(value)) {
				return asSerializableArray(value);
			}

			return value;
		});
	}

	function axisTitle(layout, axisName) {
		const title = layout?.[axisName]?.title;
		return typeof title === "string" ? title : title?.text || null;
	}

	function makeJson(graphDiv) {
		const layout = graphDiv._fullLayout || graphDiv.layout || {};
		const title =
			graphDiv.layout?.title?.text ||
			layout.title?.text ||
			null;

		const traces = graphDiv.data
			.map((trace, index) => ({trace, index}))
			.filter(({trace}) => trace.visible !== "legendonly")
			.map(({trace, index}) => {
				const exportedTrace = {
					index: index,
					name: trace.name || `trace_${index}`,
					type: trace.type || "scatter",
					x: asSerializableArray(trace.x),
					y: asSerializableArray(trace.y),
				};

				if (trace.z !== undefined) {
					exportedTrace.z = asSerializableArray(trace.z);
				}

				return exportedTrace;
			});

		return JSON.stringify({
			format: "plotly-traces",
			version: 1,
			title: title,
			xaxis_title: axisTitle(layout, "xaxis"),
			yaxis_title: axisTitle(layout, "yaxis"),
			traces: traces,
		}, null, 2);
	}

	function filenameFor(graphDiv) {
		const title =
			graphDiv.layout?.title?.text ||
			graphDiv._fullLayout?.title?.text ||
			"plot_trace";

		const plainTitle = title
			.replace(/<[^>]*>/g, "")
			.replace(/[^a-zA-Z0-9_-]+/g, "_")
			.replace(/^_+|_+$/g, "")
			.toLowerCase();

		return `${plainTitle || "plot_trace"}.json`;
	}

	function downloadTrace(graphDiv) {
		const json = makeJson(graphDiv);
		const blob = new Blob([json], {
			type: "application/json;charset=utf-8",
		});

		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");

		link.href = url;
		link.download = filenameFor(graphDiv);
		document.body.appendChild(link);
		link.click();
		link.remove();

		URL.revokeObjectURL(url);
	}

	function addDownloadButton(graphDiv) {
		const modebar = graphDiv.querySelector(".modebar");

		if (!modebar || modebar.querySelector(`.${BUTTON_CLASS}`)) {
			return;
		}

		const group =
			modebar.querySelector(".modebar-group:last-child") ||
			modebar;

		const button = document.createElement("a");
		button.className = `modebar-btn ${BUTTON_CLASS}`;
		button.title = "Download displayed trace as JSON";
		button.setAttribute("aria-label", button.title);
		button.setAttribute("role", "button");

		button.innerHTML = `
			<svg viewBox="0 0 24 24"
				 width="1em"
				 height="1em"
				 aria-hidden="true">
				<path
					fill="currentColor"
					d="M5 20h14v-2H5v2zm14-9h-4V3H9v8H5l7 7 7-7z">
				</path>
			</svg>
		`;

		button.addEventListener("click", event => {
			event.preventDefault();
			event.stopPropagation();
			downloadTrace(graphDiv);
		});

		group.appendChild(button);
	}

	function scanGraphs() {
		document
			.querySelectorAll(".js-plotly-plot")
			.forEach(addDownloadButton);
	}

	let scanScheduled = false;

	const observer = new MutationObserver(() => {
		if (scanScheduled) {
			return;
		}

		scanScheduled = true;

		window.requestAnimationFrame(() => {
			scanScheduled = false;
			scanGraphs();
		});
	});

	function initialise() {
		scanGraphs();

		observer.observe(document.body, {
			childList: true,
			subtree: true,
		});
	}

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", initialise);
	} else {
		initialise();
	}
})();
