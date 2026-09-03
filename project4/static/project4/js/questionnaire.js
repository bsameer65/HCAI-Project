(function () {
  "use strict";

  const form = document.querySelector("[data-questionnaire-form]");
  if (!form) {
    return;
  }

  const responseTimeField = form.querySelector("[data-response-time]");
  const questionnaireStartedAt = performance.now();

  form.addEventListener("submit", function () {
    responseTimeField.value = Math.round(
      performance.now() - questionnaireStartedAt
    );
  });
})();
