(function () {
  "use strict";

  const form = document.querySelector("[data-ranking-form]");
  const list = document.querySelector("[data-ranking-list]");
  if (!form || !list) {
    return;
  }

  const responseTimeField = form.querySelector("[data-response-time]");
  const status = form.querySelector("[data-ranking-status]");
  const taskStartedAt = performance.now();
  let draggedItem = null;

  function items() {
    return Array.from(list.querySelectorAll("[data-ranking-item]"));
  }

  function movieTitle(item) {
    return item.querySelector(".p4-ranking-movie strong").textContent.trim();
  }

  function updatePositions(announcement) {
    const currentItems = items();
    currentItems.forEach(function (item, index) {
      item.querySelector("[data-rank-number]").textContent = index + 1;
      item.querySelector('[data-move="up"]').disabled = index === 0;
      item.querySelector('[data-move="down"]').disabled =
        index === currentItems.length - 1;
    });

    if (announcement) {
      status.textContent = announcement;
    }
  }

  list.addEventListener("click", function (event) {
    const control = event.target.closest("[data-move]");
    if (!control) {
      return;
    }

    const item = control.closest("[data-ranking-item]");
    const direction = control.dataset.move;
    if (direction === "up" && item.previousElementSibling) {
      list.insertBefore(item, item.previousElementSibling);
    } else if (direction === "down" && item.nextElementSibling) {
      list.insertBefore(item.nextElementSibling, item);
    }

    const newPosition = items().indexOf(item) + 1;
    updatePositions(movieTitle(item) + " moved to position " + newPosition + ".");
    control.focus();
  });

  list.addEventListener("dragstart", function (event) {
    const item = event.target.closest("[data-ranking-item]");
    if (!item) {
      return;
    }

    draggedItem = item;
    item.classList.add("p4-ranking-row-dragging");
    event.dataTransfer.effectAllowed = "move";
  });

  list.addEventListener("dragover", function (event) {
    if (!draggedItem) {
      return;
    }

    event.preventDefault();
    const otherItems = items().filter(function (item) {
      return item !== draggedItem;
    });
    const insertBefore = otherItems.reduce(
      function (closest, item) {
        const box = item.getBoundingClientRect();
        const offset = event.clientY - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) {
          return { offset: offset, item: item };
        }
        return closest;
      },
      { offset: Number.NEGATIVE_INFINITY, item: null }
    ).item;

    if (insertBefore) {
      list.insertBefore(draggedItem, insertBefore);
    } else {
      list.appendChild(draggedItem);
    }
  });

  list.addEventListener("dragend", function () {
    if (!draggedItem) {
      return;
    }

    const title = movieTitle(draggedItem);
    const newPosition = items().indexOf(draggedItem) + 1;
    draggedItem.classList.remove("p4-ranking-row-dragging");
    draggedItem = null;
    updatePositions(title + " moved to position " + newPosition + ".");
  });

  form.addEventListener("submit", function () {
    responseTimeField.value = Math.round(performance.now() - taskStartedAt);
  });

  updatePositions();
})();
