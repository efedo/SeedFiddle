"""Recorded batch boundaries and exact row QC links, never current artifact proof."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from seedvision.benchmarking.recipes import RecipeIdentity
from seedvision.export import method_qc as qc, recipe_attempt_qc as attempts

VERSION = 3
MAX_ROWS = 32
MAX_LEDGER_BYTES = 64 * 1024 * 1024
BATCH_EXECUTOR = 'seedvision.export.project_native_graphs.export_project_native_graphs'
_STATES = {'failed', 'cancelled'}
_PHASES = {'admission', 'between_rows', 'recipe_row', 'batch_publication'}
_ARCHIVES = {'not_created', 'staging_removed', 'cleanup_unknown', 'unobserved'}
_ROW_STATES = {'provided', 'executed', 'empty', 'partial', 'failed', 'not_run', 'unavailable', 'cancelled'}


def _analytical(value):
    qc._exact(value, 'recipe_id recipe_version mode backend_id backend_version recipe_sha256 graph_sha256 settings_sha256',
              'batch selected recipe identity')
    parsed = RecipeIdentity.from_dict(value | {'layout_sha256':'0'*64})
    if parsed.analytical_dict() != value:
        raise ValueError('Batch selected analytical identity is not canonical.')
    return value


def _archive_path(value, index):
    if value is not None and value != f'graph-{index:03d}/native-results.json':
        raise ValueError('Batch archive locator is not its generated relative report path.')


def embedded_row_qc(quality, *, archive_path=None):
    if not isinstance(quality, qc.MethodQC):
        raise TypeError('Batch row requires validated recorded QC.')
    raw = quality.payload_json
    return {'payload_json':raw, 'raw_sha256':sha256(raw.encode('utf-8')).hexdigest(),
            'qc_revision':quality.revision, 'archive_path':archive_path}


def validate_row_qc(row, index, *, expected_source_sha256=None):
    """Validate historical linkage only; never open a selected source/archive."""
    if type(row) is not dict or not {'recipe_identity','status','report','nodes'} <= row.keys():
        raise ValueError('Invalid native graph ledger row.')
    identity = RecipeIdentity.from_dict(row['recipe_identity']).analytical_dict()
    if type(row['status']) is not str or row['status'] not in _ROW_STATES or type(row['nodes']) is not list:
        raise ValueError('Invalid native graph row outcome.')
    value = row.get('method_qc')
    if value is None:
        return None  # Legacy or explicitly uncaptured evidence remains unknown.
    qc._exact(value, 'payload_json raw_sha256 qc_revision archive_path', 'embedded batch row QC')
    quality = qc.MethodQC(value['payload_json'])
    if quality.payload_json != value['payload_json']:
        raise ValueError('Embedded batch QC must retain its canonical published bytes.')
    if (value['raw_sha256'] != sha256(value['payload_json'].encode('utf-8')).hexdigest()
            or value['qc_revision'] != quality.revision):
        raise ValueError('Embedded batch QC bytes or revision differ from their exact linkage.')
    captured = quality.payload['identity']
    if captured['recipe_identity'] != identity:
        raise ValueError('Batch row QC belongs to a different selected recipe.')
    _archive_path(value['archive_path'], index)
    if captured['scope'] == 'whole_native_archive':
        artifact = captured['native_artifact']
        if (value['archive_path'] is None or row['report'] != value['archive_path']
                or row['status'] != artifact['status']
                or expected_source_sha256 is not None and artifact['source_sha256'] != expected_source_sha256):
            raise ValueError('Batch archived row contradicts its captured QC.')
    elif captured['scope'] == 'recipe_attempt':
        if (value['archive_path'] is not None or row['report'] is not None or row['nodes']
                or row['status'] != captured['attempt_record']['state']):
            raise ValueError('Batch attempt row cannot assert archived or completed outputs.')
    else:
        raise ValueError('Batch row requires whole-archive or recipe-attempt QC.')
    return quality


def read_project_graph_qc(path):
    """Bounded schema3 row evidence; legacy rows return None, without fresh probes."""
    payload = qc._read(path, MAX_LEDGER_BYTES)
    qc._bytes(payload, MAX_LEDGER_BYTES)
    if (type(payload) is not dict or payload.get('format') != 'seedfiddle.project-native-graphs'
            or type(payload.get('version')) is not int or payload['version'] != 3
            or type(payload.get('graphs')) is not list or len(payload['graphs']) > MAX_ROWS):
        raise ValueError('Unsupported or oversized native graph ledger.')
    qc._digest(payload['source_sha256'])
    return tuple(validate_row_qc(row, index, expected_source_sha256=payload['source_sha256'])
                 for index, row in enumerate(payload['graphs']))


def load_project_graph_qc(path, index):
    qualities = read_project_graph_qc(path)
    if type(index) is not int or not 0 <= index < len(qualities):
        raise ValueError('Select an existing native graph ledger row.')
    return qualities[index]


@dataclass(frozen=True)
class RecipeBatchRequest:
    payload_json: str

    def __post_init__(self):
        # Reuse the strict parser, depth/byte limit and detached request format.
        if type(self.payload_json) is not str or len(self.payload_json.encode('utf-8')) > qc.LIMIT:
            raise ValueError('Batch request exceeds its byte bound.')
        def pairs(rows):
            value = {}
            for key, item in rows:
                if key in value:
                    raise ValueError('Duplicate batch request field.')
                value[key] = item
            return value
        value = json.loads(self.payload_json, object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        _validate_request(value)
        object.__setattr__(self, 'payload_json', qc._bytes(value).decode('utf-8'))

    @property
    def payload(self):
        return json.loads(self.payload_json)


def _validate_request(value):
    qc._exact(value, 'batch_id destination project_image_id expected_source_sha256 requested_recipes inputs_before', 'batch request')
    attempts._validate_request({'attempt_id':value['batch_id'], 'recipe':None, 'inputs':value['inputs_before']})
    attempts._text(value['destination'], 32768, 'batch destination')
    qc._digest(value['expected_source_sha256'])
    if value['project_image_id'] is not None:
        attempts._text(value['project_image_id'], 256, 'project image identity')
    if type(value['requested_recipes']) is not list or not 1 <= len(value['requested_recipes']) <= MAX_ROWS:
        raise ValueError('Batch selected recipe inventory exceeds its bound.')
    for identity in value['requested_recipes']:
        _analytical(identity)


def capture_recipe_batch_request(recipes, image_path, destination, expected_source_sha256, project_image_id=None):
    captured = attempts.capture_recipe_attempt_request(None, {'source_image':image_path}).payload
    return RecipeBatchRequest(qc._bytes({'batch_id':captured['attempt_id'], 'destination':str(destination),
        'project_image_id':project_image_id, 'requested_recipes':[recipe.identity.analytical_dict() for recipe in recipes],
        'expected_source_sha256':expected_source_sha256,
        'inputs_before':captured['inputs']}).decode('utf-8'))


def _link(row, index):
    quality = validate_row_qc(row, index)
    embedded = row.get('method_qc')
    return {'index':index, 'recipe_identity':RecipeIdentity.from_dict(row['recipe_identity']).analytical_dict(),
        'status':row['status'], 'qc_revision':None if quality is None else quality.revision,
        'qc_sha256':None if embedded is None else embedded['raw_sha256'],
        'archive_path':None if embedded is None else embedded['archive_path'],
        'qc_recording_error':row.get('qc_recording_error')}


def recipe_batch_attempt_qc(request, rows, *, error, phase, current_row_index=None, archives_state):
    if not isinstance(request, RecipeBatchRequest) or not isinstance(error, Exception):
        raise TypeError('Batch terminal QC requires its captured request and original exception.')
    from seedvision.benchmarking.native_recipe import NativeRecipeCancelled
    from seedvision.benchmarking.native_correction_recipe import NativeCorrectionCancelled
    captured = request.payload
    state = 'cancelled' if isinstance(error, (NativeRecipeCancelled, NativeCorrectionCancelled)) else 'failed'
    record = captured | {'state':state, 'phase':phase, 'detail':str(error)[:4096] or type(error).__name__,
        'exception_type':type(error).__name__, 'invoked_executor':BATCH_EXECUTOR,
        'current_row_index':current_row_index, 'completed_rows':[_link(row, index) for index,row in enumerate(rows)],
        'archives_state':archives_state, 'publication_completed':False,
        'inputs_after':attempts._observe_inputs([(row['name'], row['locator']) for row in captured['inputs_before']])}
    identity = {'scope':'recipe_batch_attempt', 'recipe_mode':None, 'recipe_identity':None,
        'native_artifact':None, 'stage_record':None, 'recorded_inspection':None, 'node_outcomes':None,
        'measurement_eligibility':None, 'scientific_validity':None, 'batch_record':record}
    return qc.MethodQC(qc._bytes({'format':qc.FORMAT, 'version':VERSION, 'recipe':None,
        'identity':identity, 'qc_revision':sha256(qc._bytes(identity)).hexdigest()}).decode('utf-8'))


def validate_recipe_batch(payload):
    value = payload['identity']
    qc._exact(value, 'scope recipe_mode recipe_identity native_artifact stage_record recorded_inspection node_outcomes '
                    'measurement_eligibility scientific_validity batch_record', 'batch terminal QC')
    if (payload['recipe'] is not None or value['scope'] != 'recipe_batch_attempt'
            or any(value[key] is not None for key in value if key not in {'scope','batch_record'})):
        raise ValueError('Batch interruption cannot claim a selected recipe or verified outputs.')
    row = value['batch_record']
    qc._exact(row, 'batch_id destination project_image_id expected_source_sha256 requested_recipes inputs_before inputs_after state phase detail '
                  'exception_type invoked_executor current_row_index completed_rows archives_state publication_completed',
                  'batch terminal record')
    _validate_request({key:row[key] for key in ('batch_id','destination','project_image_id','expected_source_sha256','requested_recipes','inputs_before')})
    if (type(row['state']) is not str or row['state'] not in _STATES or type(row['phase']) is not str
            or row['phase'] not in _PHASES or row['invoked_executor'] != BATCH_EXECUTOR
            or type(row['archives_state']) is not str or row['archives_state'] not in _ARCHIVES
            or row['publication_completed'] is not False):
        raise ValueError('Invalid terminal batch phase, state or publication claim.')
    attempts._text(row['detail'],4096,'batch detail')
    attempts._text(row['exception_type'],256,'batch exception')
    attempts._validate_inputs(row['inputs_after'])
    if [(r['name'],r['locator']) for r in row['inputs_before']] != [(r['name'],r['locator']) for r in row['inputs_after']]:
        raise ValueError('Batch changed its selected source inventory.')
    links = row['completed_rows']
    if type(links) is not list or len(links) > len(row['requested_recipes']):
        raise ValueError('Batch row evidence exceeds its selected inventory.')
    for index, link in enumerate(links):
        qc._exact(link, 'index recipe_identity status qc_revision qc_sha256 archive_path qc_recording_error', 'batch row QC link')
        if (type(link['index']) is not int or link['index'] != index
                or link['recipe_identity'] != row['requested_recipes'][index]
                or type(link['status']) is not str or link['status'] not in _ROW_STATES):
            raise ValueError('Batch completed rows are not the exact selected prefix.')
        _archive_path(link['archive_path'], index)
        if link['qc_sha256'] is None or link['qc_revision'] is None:
            if link['qc_sha256'] is not None or link['qc_revision'] is not None or link['archive_path'] is not None:
                raise ValueError('Uncaptured batch QC cannot claim a partial linkage.')
            attempts._text(link['qc_recording_error'],4096,'row QC recording failure')
        else:
            qc._digest(link['qc_sha256']); qc._digest(link['qc_revision'])
            if link['qc_recording_error'] is not None:
                raise ValueError('Captured row QC cannot claim an uncaptured recording failure.')
    current = row['current_row_index']
    if row['phase'] == 'recipe_row':
        if (type(current) is not int or not 0 <= current < len(row['requested_recipes'])
                or len(links) not in {current,current+1}):
            raise ValueError('Batch active row is inconsistent with its completed prefix.')
    elif current is not None:
        raise ValueError('Batch outside-row boundary cannot acquire a selected row.')
    if (row['phase'] == 'admission' and links
            or row['phase'] == 'between_rows' and len(links) >= len(row['requested_recipes'])
            or row['phase'] == 'batch_publication' and len(links) != len(row['requested_recipes'])):
        raise ValueError('Batch phase contradicts its recorded row inventory.')
    if payload['qc_revision'] != sha256(qc._bytes(value)).hexdigest():
        raise ValueError('Batch QC revision differs from its captured facts.')


def validate_batch_qc_links(quality, raw_records):
    """Private history/snapshot requires every exact linked row QC document."""
    if quality.payload['identity']['scope'] != 'recipe_batch_attempt':
        return
    for link in quality.payload['identity']['batch_record']['completed_rows']:
        if link['qc_sha256'] is None:
            continue
        raw = raw_records.get(link['qc_sha256'])
        if raw is None or sha256(raw).hexdigest() != link['qc_sha256']:
            raise ValueError('Batch QC has a missing or changed exact row document.')
        linked = qc.MethodQC(raw.decode('utf-8'))
        captured = linked.payload['identity']
        if linked.revision != link['qc_revision'] or captured['recipe_identity'] != link['recipe_identity']:
            raise ValueError('Batch QC row revision or selected recipe differs.')
        if captured['scope'] == 'whole_native_archive':
            if (link['archive_path'] is None or link['status'] != captured['native_artifact']['status']
                    or captured['native_artifact']['source_sha256'] != quality.payload['identity']['batch_record']['expected_source_sha256']):
                raise ValueError('Batch archive link contradicts its captured row outcome.')
        elif (captured['scope'] != 'recipe_attempt' or link['archive_path'] is not None
                or link['status'] != captured['attempt_record']['state']):
            raise ValueError('Batch attempt link contradicts its captured row outcome.')


def load_project_native_graph_history(directory):
    """Explicit destination-owned history; never infer nearby project ownership."""
    from seedvision.persistence.method_qc_history import load_project_method_qc
    destination = Path(directory).absolute()
    history = load_project_method_qc(destination.parent, destination)
    for raw in history.records:
        quality = qc.MethodQC(raw.decode('utf-8'))
        record = quality.payload['identity'].get('batch_record')
        if record is not None and record['destination'] != str(destination):
            return replace(history, records=(), acknowledgments=(), incomplete=True,
                diagnostics=(*history.diagnostics,'Batch QC destination association differs from this selected history.'))
    return history
