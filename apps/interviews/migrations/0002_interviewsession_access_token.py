import uuid

from django.db import migrations, models


def populate_access_tokens(apps, schema_editor):
    """
    Backfills a unique access_token for every existing InterviewSession
    row before the unique constraint is enforced in the next step.
    Can't rely on the model's save() override here — data migrations
    must use the historical model, which doesn't have that logic.
    """
    InterviewSession = apps.get_model("interviews", "InterviewSession")
    for session in InterviewSession.objects.filter(access_token=""):
        session.access_token = uuid.uuid4().hex
        session.save(update_fields=["access_token"])


def noop_reverse(apps, schema_editor):
    # Nothing to reverse — going back just drops the column, handled
    # by the RemoveField reversal of the AddField below.
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("interviews", "0001_initial"),
    ]

    operations = [
        # Step 1: add the field WITHOUT uniqueness, default blank —
        # safe against existing rows since nothing is unique yet.
        migrations.AddField(
            model_name="interviewsession",
            name="access_token",
            field=models.CharField(max_length=64, editable=False, blank=True, default=""),
        ),
        # Step 2: backfill every existing row with a real unique value.
        migrations.RunPython(populate_access_tokens, noop_reverse),
        # Step 3: NOW it's safe to enforce uniqueness, since no two
        # rows share the "" default anymore.
        migrations.AlterField(
            model_name="interviewsession",
            name="access_token",
            field=models.CharField(max_length=64, unique=True, editable=False, blank=True),
        ),
    ]