(function () {
  "use strict";

  const form = document.querySelector("[data-pairwise-form]");
  if (!form) {
    return;
  }

  const responseTimeField = form.querySelector("[data-response-time]");
  const taskStartedAt = performance.now();

  form.addEventListener("submit", function () {
    responseTimeField.value = Math.round(performance.now() - taskStartedAt);
  });
})();
