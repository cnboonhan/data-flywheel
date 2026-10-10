// The tiled background (img/background.png) appears on 1 page load in 42; the rest keep Keycloak's own background.
if (Math.random() < 1 / 42) document.documentElement.classList.add("flywheel-bg");
